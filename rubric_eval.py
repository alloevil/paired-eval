# -*- coding: utf-8 -*-
"""Rubric 判定器 — score 层(非可验证质量)的最小实现。

HealthBench 式 per-instance rubric:每条 criterion 是自足、可判定的单一断言,
逐条独立判定(met/not_met/abstain),加权聚合。设计要点(来自会话方法论):
  - 一条 criterion 一次判定,防 criterion conflation;
  - judge 不确定必须 abstain,禁止硬判 —— abstain 率高说明条目写得不自足,
    该修 rubric 而不是怪 judge(rubric 本身是被评测对象);
  - 可选 reference(专家参照): reference-guided 判定远比无参照可靠;
  - score 只在 met/not_met 上算,abstain 单列。

依赖注入同 claim_eval: llm(prompt, system, schema) -> dict。
"""

JUDGE_RUBRIC_SYSTEM = """你是 rubric 单条判定器。一次只判定一条 criterion 是否被 response 满足。
verdict:
- met: response 明确满足该条
- not_met: response 未满足该条,或与之矛盾
- abstain: 无法判定(criterion 含混/response 未涉及且条目不要求必须涉及/需要外部信息)
不确定时必须 abstain,禁止硬判。若给出 reference(专家参照),用它理解"什么算满足",
但判定对象始终是 response,不是 reference。"""

JUDGE_RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["met", "not_met", "abstain"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
}


def verify_rubric(response, criteria, llm, reference=None, pmap=map):
    """criteria: [{"text": 自足断言, "weight": 正数}, ...]。逐条独立判定。"""
    if not criteria:
        raise ValueError("criteria 不能为空")
    for c in criteria:
        if not c.get("text") or not isinstance(c.get("weight"), (int, float)) \
                or isinstance(c["weight"], bool) or c["weight"] <= 0:
            raise ValueError(f"非法 criterion(需要非空 text 与正数 weight): {c!r}")
    ref = f"\n\nreference(专家参照):\n{reference}" if reference else ""

    def judge(c):
        out = llm(f"criterion: {c['text']}\n\nresponse:\n{response}{ref}",
                  JUDGE_RUBRIC_SYSTEM, JUDGE_RUBRIC_SCHEMA)
        if not isinstance(out, dict) or out.get("verdict") not in ("met", "not_met", "abstain"):
            out = {"verdict": "abstain",  # 疯输出降级弃权: 不进分母,不污染分数
                   "reasoning": f"判定器输出非法,已降级为 abstain: {str(out)[:80]}"}
        return {**c, **out}

    return list(pmap(judge, criteria))


def aggregate_rubric(results):
    """加权得分(abstain 不进分母) + abstain 按条数与按权重双计量。
    judged_weight_share 是关键诚实指标: abstain_rate 只数条数, 若被弃权的恰是高权重条目,
    分数会显示 1.00 而实际上只判了 rubric 的一小部分。例: 权重 3 的核心条弃权、
    权重 1 的边缘条命中 -> score=1.00 但 judged_weight_share=0.25, 这个分数不可用。"""
    met_w = sum(r["weight"] for r in results if r["verdict"] == "met")
    not_w = sum(r["weight"] for r in results if r["verdict"] == "not_met")
    total_w = sum(r["weight"] for r in results)
    abstain = sum(r["verdict"] == "abstain" for r in results)
    denom = met_w + not_w
    return {
        "n": len(results),
        "score": met_w / denom if denom else None,
        "abstain_rate": abstain / len(results) if results else None,
        "met": sum(r["verdict"] == "met" for r in results),
        "not_met": sum(r["verdict"] == "not_met" for r in results),
        "abstain": abstain,
        "total_weight": total_w, "judged_weight": denom,
        "judged_weight_share": denom / total_w if total_w else None,
    }


def run_rubric(response, criteria, llm, reference=None, pmap=map):
    results = verify_rubric(response, criteria, llm, reference=reference, pmap=pmap)
    return {"criteria": results, "metrics": aggregate_rubric(results)}


def rubric_canary(criteria, good_response, fooling_response, llm,
                  margin=0.2, reference=None, pmap=map):
    """rubric 上线门禁: 好回答与糊弄回答(空洞奉承/正确的废话)的得分差必须 >= margin。
    过不了 = 条目可被表面特征满足(gaming), 修 rubric 而不是继续用。
    改任何 criterion 后必须重跑 —— rubric 本身是被评测对象。"""
    g = run_rubric(good_response, criteria, llm, reference=reference, pmap=pmap)
    b = run_rubric(fooling_response, criteria, llm, reference=reference, pmap=pmap)
    gs, bs = g["metrics"]["score"], b["metrics"]["score"]
    separation = (gs or 0.0) - (bs or 0.0)
    return {"good_score": gs, "fooling_score": bs, "separation": separation,
            "passed": gs is not None and separation >= margin,
            "good": g["criteria"], "fooling": b["criteria"]}
