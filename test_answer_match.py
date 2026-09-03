# -*- coding: utf-8 -*-
"""answer_match 离线测试:纯函数,零网络。运行: python3 test_answer_match.py"""
import answer_match as am


def test_parse_number():
    assert am.parse_number("8635亿") == 8635e8
    assert am.parse_number("8,635亿元") == 8635e8
    assert am.parse_number("863,500,000,000") == 8.635e11
    assert am.parse_number("1.5万亿") == 1.5e12
    assert am.parse_number("30%") == 0.3
    assert am.parse_number("３０％") == 0.3, "全角数字与百分号"
    assert am.parse_number("1/2") == 0.5
    assert am.parse_number("-42") == -42.0
    assert am.parse_number("约1993亿元") == 1993e8, "前缀修饰词不影响"
    assert am.parse_number("没有数字") is None


def test_match_numeric():
    assert am.match_numeric("8635亿", "863,500,000,000")
    assert am.match_numeric("0.5", "1/2")
    assert am.match_numeric("30%", "0.3")
    assert not am.match_numeric("8635万", "8635亿"), "单位错四个量级必须判错"
    assert not am.match_numeric("1900亿", "2000亿", rel_tol=0.01), "5%偏差在1%容差外"
    assert am.match_numeric("1900亿", "2000亿", rel_tol=0.06), "容差是显式参数"
    assert not am.match_numeric("无", "42")


def test_match_choice_and_set():
    assert am.match_choice("Ａ．", "a"), "全角+标点归一化"
    assert am.match_choice("Beijing", ["北京", "Beijing"]), "别名表"
    assert not am.match_choice("上海", ["北京", "Beijing"])
    r = am.match_set(["a", "b", "c"], ["b", "c", "d"])
    assert abs(r["f1"] - 2 / 3) < 1e-9 and not r["exact"]
    assert am.match_set(["x", "y"], ["Y", "x"])["exact"], "顺序无关+归一化"
    assert am.match_set([], ["a"])["f1"] == 0.0


def test_extract_answer():
    assert am.extract_answer("推理过程...\n答案: 42") == "42"
    assert am.extract_answer("答案：老的\n继续...\n答案：新的") == "新的", "取最后出现"
    assert am.extract_answer(r"所以 \boxed{1993亿}") == "1993亿"
    assert am.extract_answer("只有推理没有槽位") is None
    both = "答案: 早\n后面又说 \\boxed{晚}"
    assert am.extract_answer(both) == "晚", "多种槽位取位置最靠后的"
    assert am.extract_answer(r"因此 \boxed{\frac{1}{2}}") == r"\frac{1}{2}", "嵌套花括号配平"
    assert am.extract_answer(r"旧\boxed{1}新\boxed{2+2}") == "2+2", "取最后一个boxed"
    assert am.extract_answer(r"未闭合 \boxed{1+") is None, "未闭合不提取"


def test_grade_and_aggregate():
    g1 = am.grade_answer("计算...答案: 8,635亿元", "863500000000", kind="numeric")
    assert g1["verdict"] == "correct"
    g2 = am.grade_answer("答案: 9000亿", "8635亿", kind="numeric")
    assert g2["verdict"] == "incorrect"
    g3 = am.grade_answer("我不知道,无法确定。", "8635亿", kind="numeric")
    assert g3["verdict"] == "not_attempted", "无槽位=弃答,不是答错"
    g4 = am.grade_answer("答案: Beijing", ["北京", "Beijing"])
    assert g4["verdict"] == "correct"
    agg = am.aggregate_grades([g1, g2, g3, g4])
    assert agg["attempted"] == 3 and agg["correct"] == 2
    assert abs(agg["accuracy_on_attempted"] - 2 / 3) < 1e-9
    assert abs(agg["overall_accuracy"] - 0.5) < 1e-9
    assert abs(agg["not_attempted_rate"] - 0.25) < 1e-9


def test_non_attempt_reasons():
    """not_attempted 必须细分: 空白 vs 有内容但无槽位(格式不合规) ——
    混在一起就无法区分"模型没答"与"在测指令遵循"。"""
    blank = am.grade_answer("   ", "42", kind="numeric")
    assert blank["verdict"] == "not_attempted" and blank["reason"] == "blank"
    no_slot = am.grade_answer("经过计算,结果大约是四十二左右。", "42", kind="numeric")
    assert no_slot["verdict"] == "not_attempted" and no_slot["reason"] == "no_slot"
    ok = am.grade_answer("答案: 42", "42", kind="numeric")
    assert ok["verdict"] == "correct" and ok["reason"] is None
    wrong = am.grade_answer("答案: 41", "42", kind="numeric")
    assert wrong["verdict"] == "incorrect" and wrong["reason"] is None
    agg = am.aggregate_grades([blank, no_slot, ok, wrong])
    assert agg["blank"] == 1 and agg["no_slot"] == 1
    assert abs(agg["no_slot_rate"] - 0.25) < 1e-9
    assert agg["attempted"] == 2 and abs(agg["accuracy_on_attempted"] - 0.5) < 1e-9, \
        "格式问题不该污染能力口径的分母"
    assert abs(agg["not_attempted_rate"] - 0.5) < 1e-9, "两类都仍计入总弃答率"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
