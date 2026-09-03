# -*- coding: utf-8 -*-
"""任务 schema 与验证分发 — 把 answer_match(L1) 与 claim_eval 接成单一入口。

task 是声明式 dict:
    {"id": "t1", "instruction": "...",
     "verification": {"class": "exact", "gold": "8635亿", "kind": "numeric"}}

evaluate(task, response=..., observations=..., llm=..., search=...) 按类别路由:
    exact      -> answer_match.grade_answer        (纯程序, 无LLM)
    retrieval  -> claim_eval.run_world             (需 llm + search)
    trajectory -> claim_eval.extract_claims + verify_trajectory (需 llm + observations)
    rubric     -> rubric_eval.run_rubric           (需 llm; 可选 reference 专家参照)

统一返回 {"task_id", "verification_class", "score", "verdict"|None, "metrics"|None,
"details", "verifier_fp", "cost"?}。score 语义按类别: exact=0/1; retrieval=加权precision;
trajectory=grounding_rate; rubric=加权met率(全弃权为 None)。
summarize(rows) 按类别汇总 n/均分/成本合计, 混指纹默认拒绝 —— 跨指纹分数不可混合。
repeat_evaluate(task, n=8, ...) 同配置重复运行: judge 类路由跨 run 有方差, 单 run 无结论;
二值任务附 successes, 配 claim_eval.pass_hat_k 读可靠性。
evaluate_pair(task, resp_a, resp_b, n) 交错评判两侧回答(逐轮交换先后):
judge 也会漂移且先后可能不对称, 顺序判完A再判B会把这些混进系统差异。

execution / deferred / preference / policy 不在本模块职责内 —— 它们分别需要
沙盒、时间、人类锚点、规范文本, 属宿主系统; 传入会得到明确报错而不是静默降级。
(rubric 类覆盖的是"有 per-instance 质量条目"的情形, 不等于自由 preference 评分。)
"""

import hashlib
import statistics
import time

import answer_match as am
import claim_eval as ce
import rubric_eval as re_

SUPPORTED = ("exact", "retrieval", "trajectory", "rubric")


def verifier_fingerprint():
    """全部判定 prompt 的指纹。分数只在同指纹内可比 —— 任何 prompt 改动都会换指纹,
    跨指纹比较分数无效(等于换了把尺子)。代码版本由 VCS 管,这里只盯最易漂移的 prompt 层。"""
    parts = (ce.EXTRACT_SYSTEM, ce.JUDGE_WORLD_SYSTEM, ce.JUDGE_TRAJ_SYSTEM,
             ce.DERIVED_SYSTEM, ce.REFORMULATE_SYSTEM, ce.CORROBORATE_SYSTEM,
             re_.JUDGE_RUBRIC_SYSTEM)
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:12]


def validate_task(task):
    """入库前校验。返回 verification dict;不合法直接抛 ValueError。"""
    if not isinstance(task, dict) or "id" not in task:
        raise ValueError("task 必须是含 id 的 dict")
    v = task.get("verification")
    if not isinstance(v, dict) or "class" not in v:
        raise ValueError(f"task {task['id']}: 缺 verification.class")
    if v["class"] not in SUPPORTED:
        raise ValueError(
            f"task {task['id']}: class={v['class']!r} 不支持;本模块支持 {SUPPORTED},"
            " execution/deferred/preference/policy 属宿主系统职责")
    if v["class"] == "exact" and "gold" not in v:
        raise ValueError(f"task {task['id']}: exact 类必须提供 gold")
    if v["class"] == "rubric" and not v.get("criteria"):
        raise ValueError(f"task {task['id']}: rubric 类必须提供非空 criteria")
    return v


def evaluate(task, response=None, observations=None,
             llm=None, search=None, pmap=map, max_retries=1, corroborate=False,
             meter=None):
    """单任务评测入口。依赖在调用时校验:缺什么直接报错,不静默跳过。
    meter(claim_eval.Meter): 传入则包装 llm/search 计量,结果携带本任务成本增量 cost。"""
    v = validate_task(task)
    cls = v["class"]
    base = {"task_id": task["id"], "verification_class": cls,
            "verifier_fp": verifier_fingerprint(), "measured_at": time.time()}
    if meter is not None:
        before = meter.snapshot()
        llm = meter.wrap_llm(llm) if llm is not None else None
        search = meter.wrap_search(search) if search is not None else None

    def _finish(result):
        if meter is not None:
            result["cost"] = ce.Meter.delta(meter.snapshot(), before)
        return result

    if cls == "exact":
        if response is None:
            raise ValueError(f"task {task['id']}: exact 类需要 response")
        g = am.grade_answer(response, v["gold"], kind=v.get("kind", "text"),
                            marker=v.get("marker", "答案"),
                            rel_tol=v.get("rel_tol", 0.01))
        return _finish({**base, "score": 1.0 if g["verdict"] == "correct" else 0.0,
                        "verdict": g["verdict"], "metrics": None,
                        "details": {"extracted": g["extracted"], "gold": v["gold"],
                                    "reason": g.get("reason")}})

    if cls == "retrieval":
        if response is None or llm is None or search is None:
            raise ValueError(f"task {task['id']}: retrieval 类需要 response + llm + search")
        out = ce.run_world(response, llm, search, pmap=pmap,
                           max_retries=max_retries, corroborate=corroborate)
        m = out["metrics"]
        return _finish({**base, "score": m["weighted_precision"], "verdict": None,
                        "metrics": m, "details": out["claims"],
                        # 被排除的观点/hedge 句原文一并带出: 只报数量无法抽查误标
                        "unverifiable_claims": out["unverifiable_claims"]})

    # trajectory: 从最终回答抽 claim, 逐条对轨迹 observation 做 grounding
    if cls == "trajectory":
        if response is None or llm is None or observations is None:
            raise ValueError(f"task {task['id']}: trajectory 类需要 response + llm + observations")
        policy = v.get("grounding_policy", "must_ground")
        if policy not in ("must_ground", "allow_parametric"):
            raise ValueError(f"task {task['id']}: grounding_policy 只能是 "
                             "must_ground / allow_parametric")
        claims = ce.extract_claims(response, llm)
        # 合并 claim 元数据(importance 等): 只传 text 会让加权口径失效
        results = list(pmap(
            lambda c: {**c, **ce.verify_trajectory(c["text"], observations, llm)}, claims))
        m = ce.aggregate_trajectory(results)
        m["grounding_policy"] = policy
        if policy == "allow_parametric":
            # 真但无据的 claim(参数知识)不计违规: 从分母剔除, 单独报数。
            # 实测案例: 模型补一句"字节跳动是非上市公司" —— 事实为真但不在轨迹里,
            # deep-research 类应判违规, 通用问答类不该。
            denom = m["grounded"] + m["distorted"]
            m["parametric"] = m["fabricated"]
            score = m["grounded"] / denom if denom else None
        else:
            score = m["grounding_rate"]
        return _finish({**base, "score": score, "verdict": None,
                        "metrics": m, "details": results})

    # rubric: per-instance 质量条目, 逐条 met/not_met/abstain, 加权聚合
    if response is None or llm is None:
        raise ValueError(f"task {task['id']}: rubric 类需要 response + llm")
    out = re_.run_rubric(response, v["criteria"], llm,
                         reference=v.get("reference"), pmap=pmap)
    m = out["metrics"]
    return _finish({**base, "score": m["score"], "verdict": None,
                    "metrics": m, "details": out["criteria"]})


def evaluate_batch(items, **deps):
    """items: [{"task": ..., "response": ..., "observations": ...}, ...]
    返回按输入顺序的结果列表;score 序列可直接喂 claim_eval.paired_compare 做 A/B。"""
    return [evaluate(it["task"], response=it.get("response"),
                     observations=it.get("observations"), **deps) for it in items]


def summarize(rows, strict_fp=True, max_span_s=3600):
    """批量结果汇总: 按 verification_class 分组的 n / 均分 / 成本合计。
    两道混尺检查(同源,都是"不同尺子的分数不许并排"):
    - 指纹: rows 混有不同 verifier_fp 时报错(strict_fp=False 降级为标注全部指纹)。
    - 测量窗口: measured_at 齐备时校验跨度 <= max_span_s(默认1小时)。judge 与被评系统的
      表现都会跨会话漂移(本仓库实测同一模型同题两窗得 0/8 与 8/8), 跨窗汇总会把漂移
      读成系统差异。max_span_s=None 关闭。"""
    if not rows:
        raise ValueError("rows 不能为空")
    fps = sorted({r["verifier_fp"] for r in rows})
    if len(fps) > 1 and strict_fp:
        raise ValueError(f"混合验证器指纹 {fps}: 跨指纹分数不可比,分开汇总或重跑")
    stamps = [r["measured_at"] for r in rows if "measured_at" in r]
    if max_span_s is not None and len(stamps) == len(rows) and stamps:
        span = max(stamps) - min(stamps)
        if span > max_span_s:
            raise ValueError(f"测量窗口跨度 {span:.0f}s 超过 {max_span_s}s: "
                             "表现会跨会话漂移,请重测或显式设 max_span_s=None")
    by_class = {}
    for r in rows:
        g = by_class.setdefault(r["verification_class"],
                                {"n": 0, "scores": [], "cost": {}, "reasons": {}})
        g["n"] += 1
        if r["score"] is not None:
            g["scores"].append(r["score"])
        for k, val in (r.get("cost") or {}).items():
            g["cost"][k] = g["cost"].get(k, 0) + val
        # 非作答原因(blank/no_slot)必须在批次层可见: 单行有信号而报表看不见等于没有
        reason = (r.get("details") or {}).get("reason") if isinstance(r.get("details"), dict) else None
        if reason:
            g["reasons"][reason] = g["reasons"].get(reason, 0) + 1
    classes = {}
    for cls, g in by_class.items():
        classes[cls] = {"n": g["n"], "scored": len(g["scores"]),
                        "mean_score": sum(g["scores"]) / len(g["scores"]) if g["scores"] else None,
                        "cost": g["cost"] or None,
                        "non_attempt_reasons": g["reasons"] or None}
    return {"verifier_fp": fps[0] if len(fps) == 1 else fps, "n": len(rows),
            "classes": classes}


def repeat_evaluate(task, n=8, response=None, observations=None, **deps):
    """同一 (任务,配置) 重复 n 次 —— judge 类路由跨 run 有方差,单 run 结论无效。
    既然重复的目的就是量方差, 就必须报离散度: mean 会把 [1,0,1,0] 和恒定 0.5 说成一样,
    而前者是"判定不稳定", 后者是"稳定的中等分", 两者可用性完全不同。
    返回 mean_score / score_min / score_max / score_stdev(样本标准差, n<2 为 None)
    / scored(有效分数个数); 二值分数序列附 successes 供 claim_eval.pass_hat_k 读可靠性。"""
    if n < 1:
        raise ValueError("n >= 1")
    runs = [evaluate(task, response=response, observations=observations, **deps)
            for _ in range(n)]
    scores = [r["score"] for r in runs if r["score"] is not None]
    out = {"task_id": task["id"], "n": n, "runs": runs, "scores": scores,
           "scored": len(scores),
           "mean_score": sum(scores) / len(scores) if scores else None,
           "score_min": min(scores) if scores else None,
           "score_max": max(scores) if scores else None,
           "score_stdev": statistics.stdev(scores) if len(scores) > 1 else None}
    if scores and all(s in (0.0, 1.0) for s in scores):
        out["successes"] = int(sum(scores))
    return out


def evaluate_pair(task, response_a, response_b, n=1, observations_a=None,
                  observations_b=None, **deps):
    """同一 task 下交错评判两个系统的回答: 每轮 A/B 紧邻判定, 且逐轮交换先后。
    judge 同样会跨窗漂移, 先判后判也可能不对称 —— "先把A全判完再判B"会把这两者
    混进系统差异(与 paired_bench.run_paired_repeated 同源, 那边有实测教训)。
    n>1 仅对 judge 类路由有意义(exact 类确定性)。
    返回 {"task_id","n","rows","scored_reps","dropped_reps","a_mean","b_mean","compare"}。
    任一侧分数为 None(如 rubric 全弃权)的轮次会被剔出比较, 但必须计数并归因 ——
    静默丢弃会让均值的分母悄悄变小。"""
    if n < 1:
        raise ValueError("n >= 1")
    rows = []
    for rep in range(n):
        a_first = rep % 2 == 0

        def side_a():
            return evaluate(task, response=response_a, observations=observations_a, **deps)

        def side_b():
            return evaluate(task, response=response_b, observations=observations_b, **deps)

        if a_first:
            ra = side_a()
            rb = side_b()
        else:
            rb = side_b()
            ra = side_a()
        rows.append({"rep": rep, "a": ra["score"], "b": rb["score"],
                     "a_first": a_first, "result_a": ra, "result_b": rb})
    usable = [r for r in rows if r["a"] is not None and r["b"] is not None]
    sa = [r["a"] for r in usable]
    sb = [r["b"] for r in usable]
    if not sa:
        raise ValueError("两侧均无有效分数(全部为 None),无可比数据")
    # 被剔除的轮次必须计数并归因: 静默丢弃会让 a_mean/b_mean 的分母悄悄变小
    dropped_reps = [{"rep": r["rep"],
                     "sides": [s for s, v in (("a", r["a"]), ("b", r["b"])) if v is None]}
                    for r in rows if r not in usable]
    return {"task_id": task["id"], "n": n, "rows": rows, "scored_reps": len(usable),
            "dropped_reps": dropped_reps,
            "a_mean": sum(sa) / len(sa), "b_mean": sum(sb) / len(sb),
            "compare": ce.paired_compare(sa, sb)}
