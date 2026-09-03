# -*- coding: utf-8 -*-
"""rubric_eval 离线测试:mock llm,零网络。运行: python3 test_rubric_eval.py"""
import rubric_eval as re_


def mock_llm(prompt, system, schema):
    assert system is re_.JUDGE_RUBRIC_SYSTEM
    if "条A" in prompt:
        return {"verdict": "met", "reasoning": "r"}
    if "条B" in prompt:
        return {"verdict": "not_met", "reasoning": "r"}
    if "条C" in prompt:
        return {"verdict": "abstain", "reasoning": "r"}
    if "需参照" in prompt:
        v = "met" if "REF-MARK" in prompt else "not_met"
        return {"verdict": v, "reasoning": "r"}
    return {"verdict": "abstain", "reasoning": "r"}


CRIT = [{"text": "条A", "weight": 3}, {"text": "条B", "weight": 2},
        {"text": "条C", "weight": 1}]


def test_weighted_score_and_abstain():
    out = re_.run_rubric("resp", CRIT, mock_llm)
    m = out["metrics"]
    assert abs(m["score"] - 3 / 5) < 1e-9, "abstain 的权重不进分母"
    assert abs(m["abstain_rate"] - 1 / 3) < 1e-9
    assert (m["met"], m["not_met"], m["abstain"]) == (1, 1, 1)


def test_all_abstain_and_empty():
    out = re_.run_rubric("resp", [{"text": "条C", "weight": 1}], mock_llm)
    assert out["metrics"]["score"] is None, "全弃权没有分数,不硬造0分"
    try:
        re_.run_rubric("resp", [], mock_llm)
        assert False, "空criteria应报错"
    except ValueError:
        pass


def test_reference_plumbing():
    crit = [{"text": "需参照", "weight": 1}]
    with_ref = re_.run_rubric("resp", crit, mock_llm, reference="REF-MARK 专家答案")
    without = re_.run_rubric("resp", crit, mock_llm)
    assert with_ref["metrics"]["score"] == 1.0 and without["metrics"]["score"] == 0.0, \
        "reference 必须真的送进judge输入"

def test_rubric_canary():
    def sep_llm(prompt, system, schema):
        if "全弃权" in prompt:
            return {"verdict": "abstain", "reasoning": "r"}
        v = "met" if "好回答" in prompt else "not_met"
        return {"verdict": v, "reasoning": "r"}

    crit = [{"text": "条A", "weight": 1}, {"text": "条B", "weight": 1}]
    r = re_.rubric_canary(crit, "好回答内容", "糊弄内容", sep_llm)
    assert r["passed"] and r["separation"] == 1.0
    r2 = re_.rubric_canary(crit, "糊弄内容", "糊弄内容", sep_llm)
    assert not r2["passed"], "无区分度的 rubric 必须被拦下"
    r3 = re_.rubric_canary(crit, "全弃权回答", "糊弄内容", sep_llm)
    assert not r3["passed"], "好回答全弃权 = rubric 不可判定,不许上线"



def test_criterion_validation():
    for bad in [[{"weight": 1}], [{"text": "", "weight": 1}],
                [{"text": "x", "weight": 0}], [{"text": "x", "weight": -2}],
                [{"text": "x", "weight": True}], [{"text": "x"}]]:
        try:
            re_.verify_rubric("resp", bad, mock_llm)
            assert False, f"非法criterion应拒绝: {bad}"
        except ValueError:
            pass



def test_judge_insanity_guard():
    crazy = lambda p, s, sch: ["not", "a", "dict"]
    out = re_.run_rubric("resp", [{"text": "条X", "weight": 1}], crazy)
    m = out["metrics"]
    assert m["abstain"] == 1 and m["score"] is None, "疯输出降级abstain,不污染分数"
    assert "非法" in out["criteria"][0]["reasoning"]



def test_judged_weight_share():
    """按条数的 abstain_rate 会掩盖"高权重条目被弃权": 分数看着满分,
    实际只判了 rubric 的一小部分 —— judged_weight_share 才是可用性判据。"""
    def heavy_abstains(prompt, system, schema):
        # 权重3的核心条弃权, 权重1的边缘条命中
        return {"verdict": "abstain" if "核心" in prompt else "met", "reasoning": "r"}

    crit = [{"text": "核心条", "weight": 3}, {"text": "边缘条", "weight": 1}]
    m = re_.run_rubric("resp", crit, heavy_abstains)["metrics"]
    assert m["score"] == 1.0, "剩下能判的都命中 -> 分数满分"
    assert m["abstain_rate"] == 0.5, "按条数看只是一半弃权"
    assert m["total_weight"] == 4 and m["judged_weight"] == 1
    assert m["judged_weight_share"] == 0.25, "按权重看只判了四分之一 —— 这个满分不可用"
    # 全判定时覆盖率为1
    all_met = lambda p, s, sch: {"verdict": "met", "reasoning": "r"}
    m2 = re_.run_rubric("resp", crit, all_met)["metrics"]
    assert m2["judged_weight_share"] == 1.0 and m2["score"] == 1.0
    # 全弃权: 无分数, 覆盖率为0(而非None混同"没有条目")
    all_abstain = lambda p, s, sch: {"verdict": "abstain", "reasoning": "r"}
    m3 = re_.run_rubric("resp", crit, all_abstain)["metrics"]
    assert m3["score"] is None and m3["judged_weight_share"] == 0.0




def test_canary_with_unjudgeable_fooling():
    """糊弄回答本身不可判定(全弃权 -> score=None)时, 分离度应按 0 计而非崩溃。
    此前无用例覆盖这条路径, 系统化变异把 (bs or 0.0) 改成 and 后存活。"""
    def llm(prompt, system, schema):
        # 好回答可判, 糊弄回答一律弃权
        return {"verdict": "met" if "好回答" in prompt else "abstain", "reasoning": "r"}

    crit = [{"text": "条A", "weight": 1}]
    c = re_.rubric_canary(crit, "好回答内容", "无法判定的糊弄内容", llm)
    assert c["good_score"] == 1.0 and c["fooling_score"] is None
    assert c["separation"] == 1.0, "None 按 0 计, 分离度仍可算"
    assert c["passed"] is True
    # 反向: 好回答不可判定 -> 不许上线(已有断言, 此处确认两侧None也不崩)
    all_abstain = lambda p, s, sch: {"verdict": "abstain", "reasoning": "r"}
    c2 = re_.rubric_canary(crit, "好回答内容", "糊弄内容", all_abstain)
    assert c2["good_score"] is None and c2["fooling_score"] is None
    assert c2["separation"] == 0.0 and c2["passed"] is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
