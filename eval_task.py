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

execution / deferred / preference / policy 不在本模块职责内 —— 它们分别需要
沙盒、时间、人类锚点、规范文本, 属宿主系统; 传入会得到明确报错而不是静默降级。
(rubric 类覆盖的是"有 per-instance 质量条目"的情形, 不等于自由 preference 评分。)
"""

import hashlib

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
            "verifier_fp": verifier_fingerprint()}
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
                        "details": {"extracted": g["extracted"], "gold": v["gold"]}})

    if cls == "retrieval":
        if response is None or llm is None or search is None:
            raise ValueError(f"task {task['id']}: retrieval 类需要 response + llm + search")
        out = ce.run_world(response, llm, search, pmap=pmap,
                           max_retries=max_retries, corroborate=corroborate)
        m = out["metrics"]
        return _finish({**base, "score": m["weighted_precision"], "verdict": None,
                        "metrics": m, "details": out["claims"]})

    # trajectory: 从最终回答抽 claim, 逐条对轨迹 observation 做 grounding
    if cls == "trajectory":
        if response is None or llm is None or observations is None:
            raise ValueError(f"task {task['id']}: trajectory 类需要 response + llm + observations")
        claims = ce.extract_claims(response, llm)
        results = list(pmap(
            lambda c: ce.verify_trajectory(c["text"], observations, llm), claims))
        m = ce.aggregate_trajectory(results)
        return _finish({**base, "score": m["grounding_rate"], "verdict": None,
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


def summarize(rows, strict_fp=True):
    """批量结果汇总: 按 verification_class 分组的 n / 均分 / 成本合计。
    指纹一致性是硬性检查: rows 混有不同 verifier_fp 时默认报错 —— 不同指纹 = 不同尺子,
    分数混合汇总无意义;要么分开汇总,要么统一验证器版本重跑。strict_fp=False 仅降级为标注。"""
    if not rows:
        raise ValueError("rows 不能为空")
    fps = sorted({r["verifier_fp"] for r in rows})
    if len(fps) > 1 and strict_fp:
        raise ValueError(f"混合验证器指纹 {fps}: 跨指纹分数不可比,分开汇总或重跑")
    by_class = {}
    for r in rows:
        g = by_class.setdefault(r["verification_class"],
                                {"n": 0, "scores": [], "cost": {}})
        g["n"] += 1
        if r["score"] is not None:
            g["scores"].append(r["score"])
        for k, val in (r.get("cost") or {}).items():
            g["cost"][k] = g["cost"].get(k, 0) + val
    classes = {}
    for cls, g in by_class.items():
        classes[cls] = {"n": g["n"], "scored": len(g["scores"]),
                        "mean_score": sum(g["scores"]) / len(g["scores"]) if g["scores"] else None,
                        "cost": g["cost"] or None}
    return {"verifier_fp": fps[0] if len(fps) == 1 else fps, "n": len(rows),
            "classes": classes}


def repeat_evaluate(task, n=8, response=None, observations=None, **deps):
    """同一 (任务,配置) 重复 n 次 —— judge 类路由跨 run 有方差,单 run 结论无效。
    返回逐 run 结果与聚合: mean_score(pass@1)、二值任务附 successes 与 pass^k 素材
    (可靠性用 claim_eval.pass_hat_k(successes, n, k) 计算)。"""
    if n < 1:
        raise ValueError("n >= 1")
    runs = [evaluate(task, response=response, observations=observations, **deps)
            for _ in range(n)]
    scores = [r["score"] for r in runs if r["score"] is not None]
    out = {"task_id": task["id"], "n": n, "runs": runs, "scores": scores,
           "mean_score": sum(scores) / len(scores) if scores else None}
    if scores and all(s in (0.0, 1.0) for s in scores):
        out["successes"] = int(sum(scores))
    return out
