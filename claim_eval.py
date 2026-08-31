# -*- coding: utf-8 -*-
"""claim 验证流水线 — 最小可用实现。

零第三方依赖。两个外部能力全部依赖注入:
    llm(prompt: str, system: str, schema: dict) -> dict   # 带 JSON-schema 约束的结构化调用
    search(query: str) -> str                              # 返回检索结果文本(开放世界验证才需要)

验证路径:
    run_world(text, llm, search)          # 开放世界: 抽取->检索->三值判定->重试->可选复核
    run_trajectory(claims, obs, llm)      # agent 轨迹: grounded / distorted / fabricated
    verify_derived(claim, obs, llm)       # 数值推导类 claim: LLM 找算式, 程序重算判定
门禁:
    selfcheck(cases, llm, search)         # 植入错误测查全(recall) + 干净文档测误报(fp_rate)
统计:
    wilson_ci(k, n)                       # 比例的 95% 置信区间
    paired_compare(scores_a, scores_b)    # 同任务配对 A/B: 置换检验 p 值 + bootstrap CI
    required_tasks(p_win, p_loss)         # 功效分析: 要多少任务才能检出这个差距
    pass_hat_k(successes, n, k)           # 可靠性: n 次运行中任取 k 次全过的概率
适配:
    Meter / make_resilient / throttled_pmap   # 成本计量 / 有界重试 / 限并发
"""

import math
import random
import re
import time

# ---------------------------------------------------------------- prompts

EXTRACT_SYSTEM = """你是事实核查流水线的 claim 抽取器。把输入文本分解为原子 claim,规则:
1. 一条 claim = 一个可独立判真伪的陈述(单一谓词)。复合句拆开。
2. 去语境化:每条 claim 必须自足——指代消解(他/该公司→具体名字)、时间锚定(现任→截至某年)。
3. 只输出可验证的事实陈述;观点、修辞、hedge 句标 verifiable=false。
4. 语义重复只保留一条。
5. search_query 必须中性:只含主体和属性,禁止包含 claim 声称的具体值(年份、数字、结论)。
6. importance: core=核心结论 / supporting=支撑事实 / detail=边缘细节。"""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "source_quote": {"type": "string"},
            "verifiable": {"type": "boolean"},
            "importance": {"type": "string", "enum": ["core", "supporting", "detail"]},
            "search_query": {"type": "string"},
        },
        "required": ["text", "source_quote", "verifiable", "importance", "search_query"],
    }}},
    "required": ["claims"],
}

JUDGE_WORLD_SYSTEM = """你是事实核查判定器。只根据给出的检索证据判定 claim,不使用你自己的知识。
verdict:
- supported: 证据明确支持 claim
- contradicted: 证据明确与 claim 矛盾(必须引用矛盾的证据原文)
- insufficient: 证据不足(没提到/只有间接信息/来源互相矛盾)
禁止脑补。证据没提到就是 insufficient。"""

JUDGE_WORLD_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "contradicted", "insufficient"]},
        "evidence_quote": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "evidence_quote", "reasoning"],
}

JUDGE_TRAJ_SYSTEM = """你是 agent 轨迹忠实性判定器。只根据工具 observation 判定 claim,不用自己的知识。
verdict:
- grounded: 某条 observation 明确蕴含该 claim
- distorted: observation 提到相关内容,但 claim 歪曲/夸大/改动了它(数字不符、丢限定词、以偏概全)
- fabricated: 所有 observation 都不含该 claim 的依据
给出依据的 tool_call_id(fabricated 为 null)与原文引用。"""

JUDGE_TRAJ_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["grounded", "distorted", "fabricated"]},
        "source_tool_call": {"type": ["string", "null"]},
        "evidence_quote": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "source_tool_call", "evidence_quote", "reasoning"],
}

REFORMULATE_SYSTEM = """上一轮检索证据不足。为这条 claim 重写一个不同角度的中性搜索词:
换关键词组合或上位概念,不含 claim 声称的具体值。只返回搜索词。"""

REFORMULATE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

CORROBORATE_SYSTEM = """已有一轮证据与 claim 矛盾。为独立复核这一矛盾,重写一个不同角度的
中性搜索词:换关键词组合或信息源角度,不含 claim 声称的具体值。只返回搜索词。"""

DERIVED_SYSTEM = """你是数值推导审计器。给你一条含数字的 claim 和工具 observations:
判断 claim 所描述的那个"量"(如增量、比率、总和)能否由 observations 中出现的数字经算术推导得出。
- computable 只看输入是否凑得齐,严禁判断 claim 声称值的对错:即使你算出的结果与声称值明显不同,
  只要该量可推导,computable 仍为 true——对错由程序重算判定,不是你的职责。
- computable=true 时给出 expression:只能包含 observations 中出现的数字和 + - * / ( ) 空格,
  统一原始单位(亿元→数字本身,百分比→小数)。列出每个输入数字及其来源 tool_call_id。
- claim 声称的数值原样填入 claimed_value(同单位),不做修正。
- observations 中凑不齐输入时 computable=false。"""

DERIVED_SCHEMA = {
    "type": "object",
    "properties": {
        "computable": {"type": "boolean"},
        "expression": {"type": "string"},
        "claimed_value": {"type": "number"},
        "inputs": {"type": "array", "items": {"type": "object", "properties": {
            "value": {"type": "number"}, "tool_call_id": {"type": "string"}},
            "required": ["value", "tool_call_id"]}},
    },
    "required": ["computable"],
}

# ---------------------------------------------------------------- stages

WEIGHTS = {"core": 3, "supporting": 2, "detail": 1}
EVIDENCE_CAP = 6000  # 证据文本截断长度


def _safe_verdict(out, allowed, fallback=None):
    """判定器输出防疯: 非 dict 或 verdict 不在合法枚举内时,有 fallback 则降级(带留痕),
    否则响亮报错。每一处消费 LLM 输出的代码都必须假设对方随时可能发疯。"""
    if isinstance(out, dict) and out.get("verdict") in allowed:
        return out
    if fallback is not None:
        return {"verdict": fallback, "evidence_quote": "",
                "reasoning": f"判定器输出非法,已降级为 {fallback}: {str(out)[:80]}"}
    raise ValueError(f"判定器输出非法: {str(out)[:120]}")


def _extract_all(text, llm):
    """抽取全部条目(含 verifiable=false 的观点/hedge 句)。输出形状非法时响亮报错。"""
    out = llm(f"分解以下文本:\n\n{text}", EXTRACT_SYSTEM, EXTRACT_SCHEMA)
    claims = out.get("claims") if isinstance(out, dict) else None
    if not isinstance(claims, list):
        raise ValueError(f"抽取器输出非法(缺 claims 列表): {str(out)[:120]}")
    for c in claims:
        if not isinstance(c, dict) or not c.get("text") \
                or not isinstance(c.get("verifiable"), bool) \
                or c.get("importance") not in WEIGHTS or not c.get("search_query"):
            raise ValueError(f"抽取器输出非法 claim: {str(c)[:120]}")
    return claims


def extract_claims(text, llm):
    """Stage 1-3: 分解 + 去语境化 + 可验证性筛选,只返回可验证条目。"""
    return [c for c in _extract_all(text, llm) if c["verifiable"]]


def verify_world(claim, llm, search, max_retries=1, corroborate=False):
    """Stage 4: 检索 + 三值判定; insufficient 时 query 重构重试。
    corroborate=True: contradicted 必须换角度二次检索复核 —— 两轮都矛盾才定罪,
    复核不一致降级为 insufficient(来源冲突)。单一来源定罪是在测检索运气,生产建议开启。"""
    query = claim["search_query"]
    for attempt in range(max_retries + 1):
        evidence = str(search(query))[:EVIDENCE_CAP]
        verdict = _safe_verdict(llm(
            f"claim: {claim['text']}\n\n检索证据:\n{evidence}",
            JUDGE_WORLD_SYSTEM, JUDGE_WORLD_SCHEMA,
        ), {"supported", "contradicted", "insufficient"}, fallback="insufficient")
        if verdict["verdict"] != "insufficient" or attempt == max_retries:
            break
        out = llm(f"claim: {claim['text']}\n上次搜索词: {query}",
                  REFORMULATE_SYSTEM, REFORMULATE_SCHEMA)
        query = (out.get("query") if isinstance(out, dict) else None) or query
    result = {**claim, **verdict, "retries": attempt}
    if corroborate and verdict["verdict"] == "contradicted":
        out2 = llm(f"claim: {claim['text']}\n上次搜索词: {query}",
                   CORROBORATE_SYSTEM, REFORMULATE_SCHEMA)
        q2 = (out2.get("query") if isinstance(out2, dict) else None) or query
        ev2 = str(search(q2))[:EVIDENCE_CAP]
        v2 = _safe_verdict(llm(f"claim: {claim['text']}\n\n检索证据:\n{ev2}",
                               JUDGE_WORLD_SYSTEM, JUDGE_WORLD_SCHEMA),
                           {"supported", "contradicted", "insufficient"},
                           fallback="insufficient")
        if v2["verdict"] == "contradicted":
            result["corroborated"] = True
        else:
            result.update(verdict="insufficient", corroborated=False,
                          reasoning=f"单源矛盾未获复核(复核轮={v2['verdict']}): "
                                    f"{result.get('reasoning', '')}")
    return result


def verify_trajectory(claim_text, observations, llm):
    """轨迹内 grounding 判定。observations: [{tool_call_id, tool, observation}, ...]"""
    obs_text = "\n".join(
        f"[{o['tool_call_id']}] ({o.get('tool', '?')}) {o['observation']}" for o in observations
    )
    verdict = _safe_verdict(
        llm(f"claim: {claim_text}\n\n轨迹observations:\n{obs_text}",
            JUDGE_TRAJ_SYSTEM, JUDGE_TRAJ_SCHEMA),
        {"grounded", "distorted", "fabricated"})  # 无中性桶: 非法输出响亮报错
    return {"text": claim_text, **verdict}


_EXPR_OK = re.compile(r"[\d\s\.\+\-\*/\(\)]+")


def verify_derived(claim_text, observations, llm, rel_tol=0.02):
    """数值推导类 claim: LLM 只负责找出算式与输入来源,数值判定由程序重算完成(不经 judge)。"""
    obs_text = "\n".join(f"[{o['tool_call_id']}] {o['observation']}" for o in observations)
    plan = llm(f"claim: {claim_text}\n\nobservations:\n{obs_text}", DERIVED_SYSTEM, DERIVED_SCHEMA)
    if not plan.get("computable"):
        return {"text": claim_text, "verdict": "not-derivable", **plan}
    expr = plan.get("expression", "")
    if not expr or not _EXPR_OK.fullmatch(expr):
        return {"text": claim_text, "verdict": "bad-expression", **plan}
    try:
        value = eval(expr, {"__builtins__": {}}, {})  # 白名单正则已限定只含数字与算符
    except Exception as exc:                          # 语法非法/除零等: judge输出不可信,不许崩批
        return {"text": claim_text, "verdict": "bad-expression",
                "error": f"{type(exc).__name__}: {exc}", **plan}
    claimed = plan.get("claimed_value")
    if not isinstance(claimed, (int, float)) or isinstance(claimed, bool):
        return {"text": claim_text, "verdict": "bad-expression",
                "error": f"claimed_value 非法: {claimed!r}", **plan}
    ok = abs(value - claimed) <= rel_tol * max(abs(value), abs(claimed), 1e-9)
    return {"text": claim_text, "verdict": "derived-ok" if ok else "derived-wrong",
            "recomputed": value, **plan}


# ---------------------------------------------------------------- aggregate

def wilson_ci(k, n, z=1.96):
    """比例 k/n 的 Wilson 95% 置信区间。n=0 返回 None。
    小样本评测只报点估计会撒谎 —— 区间宽度本身就是"任务数不够"的警报。"""
    if not 0 <= k <= n:
        raise ValueError("需要 0 <= k <= n")
    if not n:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pass_hat_k(successes, n, k):
    """τ-bench 式 pass^k 的无偏估计: 从 n 次独立运行(其中 successes 次通过)中
    任取 k 次全部通过的概率 = C(s,k)/C(n,k)。衡量可靠性而非能力上限:
    pass@1 高但 pass^k 低 = 能做对但不稳定。精确组合数,非 (s/n)^k 近似。"""
    if not 0 <= successes <= n or not 1 <= k <= n:
        raise ValueError("需要 0 <= successes <= n 且 1 <= k <= n")
    if successes < k:
        return 0.0
    return math.comb(successes, k) / math.comb(n, k)


def aggregate_world(results):
    sup = [r for r in results if r["verdict"] == "supported"]
    con = [r for r in results if r["verdict"] == "contradicted"]
    ins = [r for r in results if r["verdict"] == "insufficient"]
    wp = lambda rs: sum(WEIGHTS[r["importance"]] for r in rs)
    denom, wdenom = len(sup) + len(con), wp(sup) + wp(con)
    return {
        "n": len(results), "supported": len(sup), "contradicted": len(con),
        "insufficient": len(ins),
        "precision": len(sup) / denom if denom else None,
        "precision_ci": wilson_ci(len(sup), denom),
        "weighted_precision": wp(sup) / wdenom if wdenom else None,
        "insufficient_rate": len(ins) / len(results) if results else None,
    }


def aggregate_trajectory(results):
    n = len(results) or 1
    count = lambda v: sum(r["verdict"] == v for r in results)
    return {
        "n": len(results),
        "grounding_rate": count("grounded") / n,
        "distortion_rate": count("distorted") / n,
        "fabrication_rate": count("fabricated") / n,
    }


def paired_compare(scores_a, scores_b, n_resamples=10000, seed=0):
    """同任务配对比较两套系统(模型/agent/harness A/B)。
    评对比必须配对——同任务同种子的分数差,不要比两个独立均值。
    返回: 均值差(A-B)、胜/负/平、符号翻转置换检验双侧 p 值、均值差的 bootstrap 95% CI。
    scores 可以是 0/1(pass) 或连续分(如每任务 claim precision),按任务对齐。"""
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("两组分数必须等长且非空(同任务配对)")
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    rng = random.Random(seed)
    # 符号翻转置换检验: H0 = A/B 可交换
    extreme = sum(
        abs(sum(d * rng.choice((1, -1)) for d in diffs) / n) >= abs(mean_diff) - 1e-12
        for _ in range(n_resamples)
    )
    p_value = (extreme + 1) / (n_resamples + 1)
    boots = sorted(
        sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples)
    )
    lo, hi = boots[int(0.025 * n_resamples)], boots[int(0.975 * n_resamples) - 1]
    return {
        "n": n, "mean_diff": mean_diff,
        "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "p_value": p_value, "diff_ci": (lo, hi),
    }


def required_tasks(p_win, p_loss, alpha=0.05, power=0.8, sims=500, n_max=2048, seed=0):
    """功效分析: 估算配对 A/B 需要多少任务才能以 power 的概率检出差异。
    p_win / p_loss = 每个任务上 A 胜 / 负于 B 的概率(其余平局), 要求 p_win > p_loss。
    模拟符号检验(正态近似)的功效, 几何网格搜最小 n; n_max 内达不到返回 None。
    paired_compare 不显著时, 用它回答"还要加多少任务", 而不是反复重跑碰运气。"""
    if not 0 <= p_loss < p_win <= 1 or p_win + p_loss > 1:
        raise ValueError("需要 0 <= p_loss < p_win 且 p_win + p_loss <= 1")
    # 双侧临界值: 对 Phi(z) = 1 - alpha/2 二分求逆
    target, lo_z, hi_z = 1 - alpha / 2, 0.0, 10.0
    for _ in range(60):
        mid = (lo_z + hi_z) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < target:
            lo_z = mid
        else:
            hi_z = mid
    z_crit = (lo_z + hi_z) / 2
    rng = random.Random(seed)

    def power_at(n):
        hits = 0
        for _ in range(sims):
            s = q = 0
            for _ in range(n):
                r = rng.random()
                if r < p_win:
                    s += 1; q += 1
                elif r < p_win + p_loss:
                    s -= 1; q += 1
            if q and abs(s) / math.sqrt(q) >= z_crit:
                hits += 1
        return hits / sims

    n = 8
    while n <= n_max:
        if power_at(n) >= power:
            return n
        n = max(n + 4, int(n * 1.4))
    return None


# ---------------------------------------------------------------- orchestrators

def run_world(text, llm, search, pmap=map, max_retries=1, corroborate=False):
    """开放世界全流程。pmap 可注入并行 map。
    观点/hedge 句不进 precision 分母,但单独计量 —— unverifiable_rate 高 = 模型在用
    不可验证的话填充篇幅,该指标不可见时这种行为完全隐形。"""
    all_claims = _extract_all(text, llm)
    claims = [c for c in all_claims if c["verifiable"]]
    results = list(pmap(
        lambda c: verify_world(c, llm, search, max_retries, corroborate), claims))
    n_unv = len(all_claims) - len(claims)
    metrics = aggregate_world(results)
    metrics["unverifiable"] = n_unv
    metrics["unverifiable_rate"] = n_unv / len(all_claims) if all_claims else None
    return {"claims": results, "metrics": metrics}


def run_trajectory(claim_texts, observations, llm, pmap=map):
    """agent 轨迹忠实性全流程。"""
    results = list(pmap(lambda t: verify_trajectory(t, observations, llm), claim_texts))
    return {"claims": results, "metrics": aggregate_trajectory(results)}


# ---------------------------------------------------------------- 门禁: 植入错误自检

def selfcheck(cases, llm, search, pmap=map, max_retries=1, corroborate=False):
    """cases: [{"text": 文档, "planted": [{"substr": 错误片段特征, "desc": 说明}]}]
    planted 非空 => 测查全: 命中 substr 的 claim 判 contradicted 才算检出。
    planted 为空 => 干净文档, 测查准: 任何 contradicted 都是误报。
    返回 recall 与 fp_rate —— 改 prompt/换 judge/换检索后必须重跑的回归指标。"""
    total, caught, details = 0, 0, []
    clean_n, fp, fp_details = 0, 0, []
    for case in cases:
        graded = run_world(case["text"], llm, search, pmap, max_retries, corroborate)["claims"]
        planted = case.get("planted") or []
        if not planted:
            clean_n += len(graded)
            bad = [r for r in graded if r["verdict"] == "contradicted"]
            fp += len(bad)
            fp_details += [{"claim": r["text"], "reasoning": r["reasoning"]} for r in bad]
            continue
        for p in planted:
            total += 1
            rel = [r for r in graded if p["substr"] in (r.get("source_quote", "") + r["text"])]
            ok = any(r["verdict"] == "contradicted" for r in rel)
            caught += ok
            details.append({"error": p["desc"], "caught": ok,
                            "verdicts": [r["verdict"] for r in rel]})
    return {"planted": total, "caught": caught,
            "recall": caught / total if total else None,
            "recall_ci": wilson_ci(caught, total), "details": details,
            "clean_claims": clean_n, "false_positives": fp,
            "fp_rate": fp / clean_n if clean_n else None,
            "fp_rate_ci": wilson_ci(fp, clean_n), "fp_details": fp_details}



# ---------------------------------------------------------------- 适配器参考实现

class Meter:
    """成本计量器: llm/search 调用次数与字符量(token 代价的粗代理)。
    报告三件套 = 分数 + CI + 成本;成本是一等公民不是附注。
    用法: 手动包装 wrap_llm/wrap_search,或把 meter 传给 eval_task.evaluate(meter=...)
    自动获得每任务成本增量(result["cost"])。"""

    def __init__(self):
        self.llm_calls = 0
        self.llm_chars_in = 0
        self.llm_chars_out = 0
        self.search_calls = 0

    def wrap_llm(self, llm):
        def wrapped(prompt, system, schema):
            self.llm_calls += 1
            self.llm_chars_in += len(prompt) + len(system)
            out = llm(prompt, system, schema)
            self.llm_chars_out += len(str(out))
            return out
        return wrapped

    def wrap_search(self, search):
        def wrapped(query):
            self.search_calls += 1
            return search(query)
        return wrapped

    def snapshot(self):
        return {"llm_calls": self.llm_calls, "llm_chars_in": self.llm_chars_in,
                "llm_chars_out": self.llm_chars_out, "search_calls": self.search_calls}

    @staticmethod
    def delta(after, before):
        return {k: after[k] - before[k] for k in after}


def make_resilient(llm, tries=3, backoff=2.0, sleep=time.sleep):
    """给 llm 调用加有界重试(线性退避)。judge/抽取调用必须容错 —— provider 流超时是常态。"""
    def wrapped(prompt, system, schema):
        for i in range(tries):
            try:
                return llm(prompt, system, schema)
            except Exception:
                if i == tries - 1:
                    raise
                sleep(backoff * (i + 1))
    return wrapped


def throttled_pmap(max_workers=4):
    """限并发的并行 map,注入给 run_* 的 pmap 参数。裸并发打 judge 会触发限流/流超时。"""
    from concurrent.futures import ThreadPoolExecutor

    def pmap(fn, xs):
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(fn, xs))
    return pmap