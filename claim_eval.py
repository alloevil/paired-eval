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
    detectable_effect(n, p_loss)          # 反问题: 任务集固定为 n 时最小可检出效应(MDE)
    pass_hat_k(successes, n, k)           # 可靠性: n 次运行中任取 k 次全过的概率
    mcnemar_exact(n_a_only, n_b_only)     # 二值配对: 不一致对的精确双侧检验
    holm_adjust(pvalues)                  # 多重比较: Holm-Bonferroni 控制 FWER
    interpret(compare, n_units)           # 结论翻译: null 强制附可排除范围
    p_floor(n) / min_units_for_alpha(a)   # 检验的 p 地板与所需最小单元数
    required_pairs(mean_diff, sd)          # 连续分数配对设计的样本量规划
适配:
    Meter / make_resilient / throttled_pmap   # 成本计量 / 有界重试 / 限并发
"""

import math
import random
import re
import threading
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
    复核不一致降级为 insufficient 并标 corroborated=False(来源冲突,不是无证据)。
    实测权衡(同一门禁 fixture): False -> recall 2/2, 单源即定罪, 会冤判口径分散的数字;
    True -> recall 1/2, 营收类数字二次检索无法独立确认而降级。故按用途选:
    生产打分要查准用 True; 门禁自检要查全用 False(selfcheck 默认即 False)。
    审计留痕: queries 记录实际发出的每个检索词(首轮/重构/复核), corroboration_verdict
    记录复核轮的原始判定 —— 只留最终 verdict 无法回答"这个结论是怎么来的"。"""
    query = claim["search_query"]
    queries = []          # 实际发出的检索词序列: 没有它就无法复现一条判定是怎么来的
    for attempt in range(max_retries + 1):
        queries.append(query)
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
    result = {**claim, **verdict, "retries": attempt, "queries": queries}
    if corroborate and verdict["verdict"] == "contradicted":
        out2 = llm(f"claim: {claim['text']}\n上次搜索词: {query}",
                   CORROBORATE_SYSTEM, REFORMULATE_SCHEMA)
        q2 = (out2.get("query") if isinstance(out2, dict) else None) or query
        queries.append(q2)
        ev2 = str(search(q2))[:EVIDENCE_CAP]
        v2 = _safe_verdict(llm(f"claim: {claim['text']}\n\n检索证据:\n{ev2}",
                               JUDGE_WORLD_SYSTEM, JUDGE_WORLD_SCHEMA),
                           {"supported", "contradicted", "insufficient"},
                           fallback="insufficient")
        result["corroboration_verdict"] = v2["verdict"]   # 复核轮的原始判定,不只是最终降级结果
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


def mcnemar_exact(n_a_only, n_b_only):
    """二值配对数据的 McNemar 精确检验(双侧): 只看不一致对 —— A对B错 n_a_only 次,
    A错B对 n_b_only 次。H0 下不一致对服从 Bin(n, 0.5), 精确二项尾概率 ×2 (上限1)。
    适用场景: 逐轮交错配对产出的二值结果, 比压成逐题均值再做置换检验保留更多信息;
    小样本下给出精确 p 值, 无需模拟。一致对(两侧同对/同错)不携带方向信息,故被忽略。"""
    if n_a_only < 0 or n_b_only < 0:
        raise ValueError("不一致对计数不能为负")
    n = n_a_only + n_b_only
    if n == 0:
        return 1.0   # 无不一致对: 数据不含任何方向证据
    lo = min(n_a_only, n_b_only)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm_adjust(pvalues):
    """Holm-Bonferroni 逐步降幂校正(按原顺序返回校正后 p 值)。
    k 个系统两两比较有 k(k-1)/2 次检验, 不校正则"比了十对庆祝那一对显著的"必然发生。
    比 Bonferroni 更有功效且同样严格控制 family-wise error rate; 强制单调不减。"""
    if not pvalues:
        return []
    if any(not 0.0 <= p <= 1.0 for p in pvalues):
        raise ValueError("p 值必须落在 [0,1]")
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (n - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


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
    """开放世界聚合。unconfirmed_contradictions 单列: 复核模式下被降级的单源矛盾
    会融进 insufficient, 那样"3条claim单源矛盾但未获确认"这个信号就消失了 ——
    它与"完全没证据"是不同的诊断(前者指向口径冲突或检索覆盖不足, 值得人工看)。"""
    sup = [r for r in results if r["verdict"] == "supported"]
    con = [r for r in results if r["verdict"] == "contradicted"]
    ins = [r for r in results if r["verdict"] == "insufficient"]
    unconfirmed = [r for r in ins if r.get("corroborated") is False]
    wp = lambda rs: sum(WEIGHTS[r["importance"]] for r in rs)
    denom, wdenom = len(sup) + len(con), wp(sup) + wp(con)
    return {
        "n": len(results), "supported": len(sup), "contradicted": len(con),
        "insufficient": len(ins),
        "unconfirmed_contradictions": len(unconfirmed),
        "precision": len(sup) / denom if denom else None,
        "precision_ci": wilson_ci(len(sup), denom),
        "weighted_precision": wp(sup) / wdenom if wdenom else None,
        "insufficient_rate": len(ins) / len(results) if results else None,
    }


def aggregate_trajectory(results):
    """轨迹忠实性聚合。三层输出:
    - 计数(grounded/distorted/fabricated): grounding policy 需要按计数重算分母。
    - 未加权比率: 每条 claim 等权。
    - 加权比率(仅当每条都带合法 importance 时给出): 与 world 路由的 weighted_precision
      同源 —— 核心事实被编造与边角细节被编造不该记同一笔账, 丢掉 importance
      等于让重大幻觉被大量无害细节稀释。"""
    n = len(results) or 1
    count = lambda v: sum(r["verdict"] == v for r in results)
    out = {
        "n": len(results),
        "grounded": count("grounded"), "distorted": count("distorted"),
        "fabricated": count("fabricated"),
        "grounding_rate": count("grounded") / n,
        "distortion_rate": count("distorted") / n,
        "fabrication_rate": count("fabricated") / n,
        "weighted_grounding_rate": None, "weighted_fabrication_rate": None,
        "weighted_distortion_rate": None,
    }
    if results and all(r.get("importance") in WEIGHTS for r in results):
        total_w = sum(WEIGHTS[r["importance"]] for r in results)
        wsum = lambda v: sum(WEIGHTS[r["importance"]] for r in results if r["verdict"] == v)
        out.update(total_weight=total_w,
                   weighted_grounding_rate=wsum("grounded") / total_w,
                   weighted_distortion_rate=wsum("distorted") / total_w,
                   weighted_fabrication_rate=wsum("fabricated") / total_w)
    return out


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
        # 有效对数 = 非零差值的对数。符号翻转对零差值对是恒等操作, 故它们不进入
        # 置换分布, 可达的最小 p 是 2/2^n_effective 而非 2/2^n。实测(第120轮):
        # 5 对零差 + 1 对满差, p=1.0000 恰是 p_floor(1), 与 p_floor(6)=0.031 无关。
        # interpret 默认拿它当 p 地板的基数 —— 与第119轮 McNemar 那处同一个 bug 类。
        "n_effective": n - sum(d == 0 for d in diffs),
        "p_value": p_value, "diff_ci": (lo, hi),
    }


def required_tasks(p_win, p_loss, alpha=0.05, power=0.8, sims=500, n_max=2048, seed=0,
                   detail=False, rand=None):
    """功效分析: 估算配对 A/B 需要多少任务才能以 power 的概率检出差异。
    p_win / p_loss = 每个任务上 A 胜 / 负于 B 的概率(其余平局), 要求 p_win > p_loss。
    内部用 mcnemar_exact 判定 —— 规划器与检验器必须用同一把尺子: 早先这里用符号检验的
    正态近似, 实测在规划出的 n 上精确检验只有 0.765 功效(目标 0.8), 即 n 被系统性低估。
    几何网格搜最小 n; n_max 内达不到返回 None。
    detail=True 时返回 {"n","achieved_power","power_target","alpha"} —— 网格搜索会跳过
    临界点, 实际功效常明显高于目标(实测 n=113 时达 0.907), 只返回 n 会让调用者
    无从知晓这个余量, 要重做一遍蒙特卡洛才能得到内部早已算出的数。
    paired_compare/mcnemar 不显著时, 用它回答"还要加多少任务", 而不是反复重跑碰运气。"""
    if not 0 <= p_loss < p_win <= 1 or p_win + p_loss > 1:
        raise ValueError("需要 0 <= p_loss < p_win 且 p_win + p_loss <= 1")
    rand = rand or random.Random(seed).random   # 唯一未注入的依赖曾是 RNG:
    # 蒙特卡洛函数的契约是统计性的, 小扰动被随机性吸收 —— 变异测试里这两个规划器的
    # 存活率远高于其他代码。注入确定性 rand 后, 边界(达标比较、搜索上界、计数器初值)
    # 才能被精确断言, 而不是靠"统计上大概对"。

    def power_at(n):
        hits = 0
        for _ in range(sims):
            a_only = b_only = 0
            for _ in range(n):
                r = rand()
                if r < p_win:
                    a_only += 1
                elif r < p_win + p_loss:
                    b_only += 1
            if mcnemar_exact(a_only, b_only) < alpha:
                hits += 1
        return hits / sims

    n = 8
    while n <= n_max:
        achieved = power_at(n)
        if achieved >= power:
            return {"n": n, "achieved_power": achieved, "power_target": power,
                    "alpha": alpha} if detail else n
        n = max(n + 4, int(n * 1.4))
    return {"n": None, "achieved_power": None, "power_target": power,
            "alpha": alpha} if detail else None


def required_pairs(mean_diff, sd, alpha=0.05, power=0.8, sims=200, resamples=2000,
                   n_max=512, seed=0, detail=False, gauss=None):
    """连续分数配对设计的样本量规划 —— required_tasks 的连续版本。

    为什么需要单独一支: required_tasks 只服务二值胜率(McNemar), 而 rubric/grounding
    这类打分是连续的, 检验走符号翻转置换。第116轮规划这类设计时只能手算
    (1.96*sd/Δ)^2 —— 那是"CI 恰好排除 0"的公式, 实测功效仅 0.45, 与实际使用的置换
    检验不是同一把尺子; 正确的功效公式 ((z_a/2 + z_power)*sd/Δ)^2 大 2.04 倍。

    遵循本模块既定原则: 规划器内部跑的就是 paired_compare, 而非闭式近似。代价是慢
    (每个候选 n 要跑 sims 次置换检验), 故 sims/resamples 默认比检验时小 —— 规划只需
    知道"大概多少", 精度余量由 detail 里的 achieved_power 交代。

    两个约束取较大者(这是第119/120轮教训的直接落地):
      1. 功效: 差值分布为 N(mean_diff, sd) 时, 检验能以 power 的概率给出 p < alpha
      2. p 地板: p_floor(n) < alpha —— 否则再大的效应也拿不到显著
    若 sd 为 0(差值恒定), 功效约束由地板单独决定。
    gauss 可注入确定性正态采样器(签名 gauss(mu, sigma)), 用于测试边界。
    detail=True 时返回 {"n","achieved_power","power_target","alpha","floor_n"}。"""
    if mean_diff == 0:
        raise ValueError("mean_diff 为 0 时无效应可检出")
    if sd < 0:
        raise ValueError(f"sd 不能为负: {sd}")
    floor_n = min_units_for_alpha(alpha, resamples)
    if sd == 0:
        # 差值恒定: 每对同号, 置换检验的 p 恒等于地板, 故样本量只由地板决定
        return {"n": floor_n, "achieved_power": 1.0, "power_target": power,
                "alpha": alpha, "floor_n": floor_n} if detail else floor_n
    g = gauss or random.Random(seed).gauss

    def power_at(n):
        hits = 0
        for i in range(sims):
            diffs = [g(mean_diff, sd) for _ in range(n)]
            # 每次 sim 换重采样种子: 复用同一个会让所有置换检验用完全相同的符号翻转
            # 模式, 各次 p 值高度相关, 功效估计随之偏斜(第121轮自查发现)。
            c = paired_compare(diffs, [0.0] * n, n_resamples=resamples, seed=seed + i)
            if c["p_value"] < alpha:
                hits += 1
        return hits / sims

    n = floor_n
    while n <= n_max:
        achieved = power_at(n)
        if achieved >= power:
            return {"n": n, "achieved_power": achieved, "power_target": power,
                    "alpha": alpha, "floor_n": floor_n} if detail else n
        n = max(n + 2, int(n * 1.15))   # 步长比 required_tasks 细: 连续分数的功效曲线
        # 平缓, 1.4 倍跳会在临界点附近超调很多(实测 42 已达 0.83 却被跳到 54)
    return {"n": None, "achieved_power": None, "power_target": power,
            "alpha": alpha, "floor_n": floor_n} if detail else None


def detectable_effect(n, p_loss=0.0, alpha=0.05, power=0.8, sims=400, seed=0, step=0.01,
                      detail=False, rand=None):
    """required_tasks 的反问题: 任务集固定为 n 时, 最小可检出的胜率是多少(MDE)。
    实践中任务集通常是给定的 —— 此时诚实报告需要的不是"再加多少题", 而是
    "这个任务集根本看不见小于多大的差异"。同样用 mcnemar_exact 判定, 与检验器一致。
    返回最小 p_win(在 p_loss 固定下达到 power 的胜率), 上界内不可达则返回 None。
    detail=True 时返回 {"mde","achieved_power","power_target","alpha","step"} ——
    网格步长决定精度, 达标点的实际功效通常高于目标, 这个余量不该被丢弃。"""
    if n < 1:
        raise ValueError("n >= 1")
    if not 0 <= p_loss < 1:
        raise ValueError("需要 0 <= p_loss < 1")
    rand = rand or random.Random(seed).random
    p_win = max(p_loss + step, step)
    while p_win + p_loss <= 1.0:
        hits = 0
        for _ in range(sims):
            a_only = b_only = 0
            for _ in range(n):
                r = rand()
                if r < p_win:
                    a_only += 1
                elif r < p_win + p_loss:
                    b_only += 1
            if mcnemar_exact(a_only, b_only) < alpha:
                hits += 1
        achieved = hits / sims
        if achieved >= power:
            mde = round(p_win, 10)
            return {"mde": mde, "achieved_power": achieved, "power_target": power,
                    "alpha": alpha, "step": step} if detail else mde
        p_win = round(p_win + step, 10)
    return {"mde": None, "achieved_power": None, "power_target": power,
            "alpha": alpha, "step": step} if detail else None


def p_floor(n, resamples=10000):
    """配对符号翻转检验在 n 个单元下的最小可能双侧 p —— 与效应大小无关的硬地板。
    2^n 种符号排列里, 完美分离只占两端各一种, 故 p >= 2/2^n; 重采样实现另有
    1/(resamples+1) 的下限, 取两者较大。
    实测踩坑(第115轮): 3 个案例得 Δ=+0.911 CI=[+0.893,+0.926] 却 p=0.252 —— 区间
    把 0 排除十万八千里, 检验却说"不显著", 因为 n=3 的地板就是 0.25。
    设计阶段用它定最小案例数: alpha=0.05 -> 需 n>=6(2/2^6=0.031)。
    这是 MDE 的另一面: MDE 问"能检出多小的效应", p_floor 问"再大的效应能拿到多小的 p"。
    """
    if n < 1:
        raise ValueError(f"n 必须 >=1, 得到 {n}")
    return max(2 / 2 ** n, 1 / (resamples + 1)) if n < 64 else 1 / (resamples + 1)


def min_units_for_alpha(alpha=0.05, resamples=10000):
    """满足 p_floor(n) < alpha 的最小配对单元数。设计阶段先问这个, 再谈效应量。
    严格小于, 与显著判据 p < alpha 对齐: 地板恰好等于 alpha 时显著性仍不可达。
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha 必须在 (0,1), 得到 {alpha}")
    n = 1
    while p_floor(n, resamples) >= alpha:
        n += 1
        if n > 64:
            raise ValueError(f"alpha={alpha} 小于重采样下限 {1/(resamples+1)}, "
                             f"需增大 resamples")
    return n


def interpret(compare, n_units=None, alpha=0.05, sims=400, seed=0, floor_n=None):
    """把配对比较结果翻译成可报告结论 —— null 分支强制附"能排除多大效应"。
    不这样做的后果本仓库亲身踩过: "Δ=0.000, p=1.000" 被写成"效应根本不存在",
    而那个设计只有 16 个配对单元, MDE=0.46, 实际只能排除 >=46% 的效应。
    n_units: 配对单元数(逐题或逐轮), 默认取 compare["n"]; 决定 MDE。
    floor_n: p 地板的基数, 默认等于 n_units。两者可以不同, 这是第119轮集成测试
      抓到的真 bug: 若 compare 的 p 来自 McNemar 精确检验, 其可达的最小 p 只由
      不一致对数决定(d 对全在一侧时 p = 2/2^d), 与总单元数无关。10 个单元里只有
      5 对不一致时, 地板是 0.0625 而不是 0.002 —— 前者意味着"这个检验根本到不了
      显著", 后者会被写成"未检出差异, 只能排除 >=67%"。两句话给读者的行动完全不同。
      MDE 仍按单元数算(单元越多, 出现不一致对的机会越多), 故两个基数各管一件事。
    返回 {"verdict","p_value","mean_diff","diff_ci","n_units","rules_out","p_floor","text"}:
      verdict="significant" -> 报效应量与区间
      verdict="null"        -> rules_out=该设计的 MDE(None 表示样本量下不可达,
                               即这条 null 无信息, 不能当证据用)
    """
    n = n_units if n_units is not None else compare["n"]
    p = compare["p_value"]
    # 地板基数优先级: 显式 floor_n(如 report 传 McNemar 的不一致对) > compare 自报的
    # 有效对数(置换检验的非零差值对) > 单元数。三者语义不同, 混用就是第119/120轮的 bug。
    fn = floor_n if floor_n is not None else compare.get("n_effective", n)
    fn = max(1, fn)
    out = {"p_value": p, "mean_diff": compare["mean_diff"],
           "diff_ci": compare["diff_ci"], "n_units": n, "rules_out": None,
           "p_floor": p_floor(fn) if n >= 1 else None}
    if p < alpha:
        out["verdict"] = "significant"
        out["text"] = (f"显著: Δ={compare['mean_diff']:+.3f} "
                       f"CI95=[{compare['diff_ci'][0]:+.3f},{compare['diff_ci'][1]:+.3f}] "
                       f"p={p:.4f} (n={n})")
        return out
    if n < 1:
        # 没有配对单元: 连"地板"都无从谈起, 这是无信息而非检验无力(合成报告场景)
        out["verdict"] = "null"
        out["p_floor"] = None
        out["text"] = (f"无信息的 null: n={n} 太小, 任何效应都检不出(MDE 不可达) —— "
                       f"不能当作『无差异』的证据")
        return out
    floor = out["p_floor"]
    if floor >= alpha:   # 地板等于 alpha 时显著性也不可达(判据是 p < alpha)
        # p 被样本量地板卡死: 效应再大也拿不到显著, 报"不显著"是误导
        # 三种基数各有不同的补救处方, 措辞必须点明是哪一种
        if floor_n is not None:
            basis = "不一致对"          # McNemar: 要么加轮次, 要么换能拉开差距的题
        elif "n_effective" in compare and compare["n_effective"] < n:
            basis = "非零差值对"        # 置换: 有单元但差值为零, 加轮次未必有用
        else:
            basis = "配对单元"          # 单纯样本少, 加轮次即可
        try:
            need = f"需至少 {min_units_for_alpha(alpha)} 个{basis}"
        except ValueError:
            # alpha 已触及重采样下限, 加单元无用 —— 报告函数在此不得抛错
            need = f"alpha 已触及重采样下限, 增加{basis}无用, 需提高 resamples"
        out["verdict"] = "null"
        out["rules_out"] = None
        out["p_floor"] = floor
        out["text"] = (f"检验无力: {basis}只有 {fn} 个, 最小可能 p 是 {floor:.3f} "
                       f">= alpha={alpha}, 效应再大也不可能显著"
                       f"(点估计 Δ={compare['mean_diff']:+.3f}, "
                       f"CI95=[{compare['diff_ci'][0]:+.3f},{compare['diff_ci'][1]:+.3f}]) "
                       f"—— {need}")
        return out
    mde = detectable_effect(n, alpha=alpha, sims=sims, seed=seed)   # n>=1 已由上方保证
    out["verdict"] = "null"
    out["rules_out"] = mde
    out["p_floor"] = floor
    if mde is None:
        out["text"] = (f"无信息的 null: n={n} 太小, 任何效应都检不出(MDE 不可达) —— "
                       f"不能当作『无差异』的证据")
    else:
        out["text"] = (f"未检出差异: p={p:.3f}, 该设计只能排除 >={mde:.0%} 的单方面胜率 "
                       f"(n={n}, 点估计 Δ={compare['mean_diff']:+.3f}); "
                       f"低于此的效应无法排除")
    return out


# ---------------------------------------------------------------- orchestrators

def run_world(text, llm, search, pmap=map, max_retries=1, corroborate=False):
    """开放世界全流程。pmap 可注入并行 map。
    观点/hedge 句不进 precision 分母,但单独计量 —— unverifiable_rate 高 = 模型在用
    不可验证的话填充篇幅,该指标不可见时这种行为完全隐形。
    unverifiable_claims 返回被排除的条目原文: 只报数量无法审计"抽取器是否把难验证的
    事实句误标成观点句" —— 那是绕过评分最省力的路径, 必须能被人抽查。"""
    all_claims = _extract_all(text, llm)
    claims = [c for c in all_claims if c["verifiable"]]
    unverifiable = [c for c in all_claims if not c["verifiable"]]
    results = list(pmap(
        lambda c: verify_world(c, llm, search, max_retries, corroborate), claims))
    metrics = aggregate_world(results)
    metrics["unverifiable"] = len(unverifiable)
    metrics["unverifiable_rate"] = (len(unverifiable) / len(all_claims)
                                    if all_claims else None)
    return {"claims": results, "unverifiable_claims": unverifiable, "metrics": metrics}


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


def trajectory_selfcheck(cases, llm, pmap=map):
    """轨迹忠实性检测器的门禁, 与 selfcheck 同源(那个管 world 路由, 这个管 trajectory)。
    cases: [{"observations": [...], "faithful": 忠实回答,
             "planted": [{"response": 被篡改的回答, "expect": "distorted"|"fabricated",
                          "desc": 说明}]}]
    - faithful 回答的 claim 应全部 grounded, 任何非 grounded 记误报(fp_rate)。
    - 每个 planted 回答必须至少命中一条 expect 类别的判定, 否则记漏检(recall)。
    distorted 类在真实数据上极少自然出现(忠实模型不歪曲), 因此必须靠人工篡改来测灵敏度 ——
    没有这道门禁, "零歪曲"既可能是模型忠实, 也可能是检测器瞎了, 两者无法区分。"""
    total, caught, details = 0, 0, []
    clean_n, fp, fp_details = 0, 0, []
    for case in cases:
        obs = case["observations"]
        if case.get("faithful"):
            claims = extract_claims(case["faithful"], llm)
            got = list(pmap(lambda c: verify_trajectory(c["text"], obs, llm), claims))
            clean_n += len(got)
            bad = [r for r in got if r["verdict"] != "grounded"]
            fp += len(bad)
            fp_details += [{"claim": r["text"], "verdict": r["verdict"],
                            "reasoning": r.get("reasoning", "")} for r in bad]
        for p in case.get("planted") or []:
            total += 1
            claims = extract_claims(p["response"], llm)
            got = list(pmap(lambda c: verify_trajectory(c["text"], obs, llm), claims))
            hit = [r for r in got if r["verdict"] == p["expect"]]
            caught += bool(hit)
            details.append({"error": p["desc"], "expect": p["expect"], "caught": bool(hit),
                            "verdicts": [r["verdict"] for r in got]})
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
        self._lock = threading.Lock()  # 与 throttled_pmap 并发使用时裸 += 会静默少记
        self.llm_calls = 0
        self.llm_chars_in = 0
        self.llm_chars_out = 0
        self.search_calls = 0

    def wrap_llm(self, llm):
        def wrapped(prompt, system, schema):
            with self._lock:
                self.llm_calls += 1
                self.llm_chars_in += len(prompt) + len(system)
            out = llm(prompt, system, schema)
            with self._lock:
                self.llm_chars_out += len(str(out))
            return out
        return wrapped

    def wrap_search(self, search):
        def wrapped(query):
            with self._lock:
                self.search_calls += 1
            return search(query)
        return wrapped

    def snapshot(self):
        with self._lock:
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