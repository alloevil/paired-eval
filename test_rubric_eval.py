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



if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
