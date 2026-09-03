#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复现检查脚本自身的测试: 用假 call 驱动 check_scaffold_effect 的四条判据。

为什么要测它: 这个脚本是"结论是否漂移"的唯一警报器。若它的判据写反(例如把
方向反转当成通过、把饱和当成成功), 漂移会静默通过 —— 与第75轮"门禁必须自测"同源。
不需要真实模型: 注入的 call 是纯函数, 因此可以进快速套件。
"""
import reproduce_findings as rf
import paired_bench as pb

SENS = [t for t in pb.ALL_TASKS if t["id"] in pb.SCAFFOLD_SENSITIVE]
CANON = {t["instruction"]: t["canonical"] for t in SENS}


def _canon_for(prompt):
    for ins, c in CANON.items():
        if ins in prompt:
            return c
    raise AssertionError(f"提示未匹配任何敏感题: {prompt[:60]}")


def strict_only(prompt):
    """严格脚手架下正确, 裸指令下失败 —— 记录中的真实形态。"""
    return _canon_for(prompt) if prompt.startswith(rf.STRICT) else "随便说点什么"


def test_reproduces_recorded_effect():
    assert rf.check_scaffold_effect(strict_only, n=4, verbose=False) == []


def test_detects_vanished_effect():
    """效应消失(两侧都对)必须报 FAIL: 既是效应量不足, 也是全饱和。"""
    problems = rf.check_scaffold_effect(lambda p: _canon_for(p), n=4, verbose=False)
    assert problems, "效应消失时必须报警"
    joined = " ".join(problems)
    assert "效应量" in joined and "已漂移" in joined
    assert "饱和" in joined and "换题而非改阈值" in joined, \
        "饱和时的处方必须是换题, 不是放宽阈值"


def test_detects_direction_reversal():
    """方向反转必须单独报出并明示"这是重大发现, 不要当失败处理"。"""
    flipped = lambda p: ("随便说点什么" if p.startswith(rf.STRICT) else _canon_for(p))
    problems = rf.check_scaffold_effect(flipped, n=4, verbose=False)
    joined = " ".join(problems)
    assert "方向反转" in joined and "重大发现" in joined, joined
    # 反转时效应量仍然大, 故不应报效应量不足 —— 报错要指向真实原因
    assert "已漂移" not in joined, f"方向反转不该被误报成漂移: {joined}"


def test_detects_weakened_significance():
    """效应变弱到不显著: 只报显著性/效应量, 不报方向问题。"""
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if prompt.startswith(rf.STRICT):
            return _canon_for(prompt) if calls["n"] % 3 else "随便说点什么"
        return "随便说点什么" if calls["n"] % 4 else _canon_for(prompt)

    problems = rf.check_scaffold_effect(flaky, n=2, verbose=False)
    assert "方向反转" not in " ".join(problems)


def test_main_returns_nonzero_on_drift():
    assert rf.main(strict_only) == 0
    assert rf.main(lambda p: _canon_for(p)) == 1
    assert rf.main() == 2, "未注入 call 时应返回 2 而不是假装通过"


def test_expected_thresholds_are_below_recorded():
    """阈值必须严格宽于记录值, 否则正常噪声就会触发假警报。"""
    assert rf.EXPECTED["scaffold_effect_min"] < 0.625
    assert rf.EXPECTED["scaffold_p_max"] > 0.0117
    assert rf.EXPECTED["informative_min"] <= 4




def _fixed(effect, p, informative, direction="strict"):
    """构造一个 report 返回, 用于精确钉死判据的边界(不经真实模型)。"""
    sign = +1.0 if direction == "bare" else -1.0   # a=bare, b=strict(字典序)
    return {"pairs": [{"a": "bare", "b": "strict", "mean_diff": sign * effect,
                       "p_mcnemar_holm": p, "concentration": 0.5,
                       "diff_ci": (sign * effect, sign * effect)}],
            "saturation": {"informative": informative, "n_tasks": 4},
            "text": "(合成)"}


def test_threshold_boundaries_are_inclusive(monkey=None):
    """恰好等于阈值必须算通过 —— 否则正常噪声会在边界上随机报警。
    三条阈值各自钉一次: 效应量 >= min 通过, p <= max 通过, 有效样本 >= min 通过。"""
    real = rf.pb.report
    try:
        E = rf.EXPECTED
        cases = [
            # (效应量, p, 有效样本, 期望问题数, 说明)
            (E["scaffold_effect_min"], E["scaffold_p_max"], E["informative_min"], 0,
             "三项恰好在阈值上 -> 全部通过"),
            (E["scaffold_effect_min"] - 1e-9, E["scaffold_p_max"], E["informative_min"], 1,
             "效应量差一点点 -> 只报效应量"),
            (E["scaffold_effect_min"], E["scaffold_p_max"] + 1e-9, E["informative_min"], 1,
             "p 超一点点 -> 只报显著性"),
            (E["scaffold_effect_min"], E["scaffold_p_max"], E["informative_min"] - 1, 1,
             "有效样本少一个 -> 只报饱和"),
        ]
        for effect, p, info, want, desc in cases:
            rf.pb.report = lambda *a, **k: _fixed(effect, p, info)
            got = rf.check_scaffold_effect(lambda p_: "x", n=1, verbose=False)
            assert len(got) == want, f"{desc}: 期望 {want} 项, 得 {len(got)}: {got}"
        # 效应量为 0 时方向判定不得反转(mean_diff == 0 走 else 分支 -> b == strict)
        rf.pb.report = lambda *a, **k: _fixed(0.0, 1.0, 4)
        got = rf.check_scaffold_effect(lambda p_: "x", n=1, verbose=False)
        assert not any("方向反转" in g for g in got), f"Δ=0 不该判方向反转: {got}"
    finally:
        rf.pb.report = real




def test_thresholds_stay_meaningful():
    """阈值要宽于记录值以容纳噪声, 但不能宽到失去意义 —— 双侧都要约束。
    反例: p_max 放到 1.05 时"任何 p 都算显著", 警报器就永远不会响。"""
    E = rf.EXPECTED
    assert 0.0117 < E["scaffold_p_max"] <= 0.05, \
        "显著性阈值必须仍是显著性水平(<=0.05), 不能宽到接受任何 p"
    assert 0 < E["scaffold_effect_min"] < 0.625, "效应量阈值必须严格介于 0 与记录值之间"
    # 有效样本下界: 必须容许真实复跑观察到的 3/4, 也要容许再饱和一题(2/4);
    # 否则一题偶然饱和就会把复现判成失败。
    real = rf.pb.report
    try:
        for info in (2, 3, 4):
            rf.pb.report = lambda *a, _i=info, **k: _fixed(0.625, 0.0117, _i)
            got = rf.check_scaffold_effect(lambda p_: "x", n=1, verbose=False)
            assert got == [], f"有效样本 {info}/4 应通过, 却报: {got}"
        rf.pb.report = lambda *a, **k: _fixed(0.625, 0.0117, 1)
        got = rf.check_scaffold_effect(lambda p_: "x", n=1, verbose=False)
        assert len(got) == 1 and "饱和" in got[0], f"仅 1/4 有信息应报饱和: {got}"
    finally:
        rf.pb.report = real


def test_task_floor_accepts_exactly_three():
    """敏感子集的下界是 3 题(记录值 4 题)。恰好 3 题必须仍能跑 —— 若写成 >3,
    子集因故缩到 3 题时脚本会拒跑, 而那时恰恰最需要它报告漂移。"""
    real_ids = pb.SCAFFOLD_SENSITIVE
    try:
        pb.SCAFFOLD_SENSITIVE = real_ids[:3]
        assert rf.check_scaffold_effect(strict_only, n=4, verbose=False) == []
        pb.SCAFFOLD_SENSITIVE = real_ids[:2]
        try:
            rf.check_scaffold_effect(strict_only, n=4, verbose=False)
            raise AssertionError("2 题应被拒跑")
        except AssertionError as e:
            assert "缩水到 2 题" in str(e), e
    finally:
        pb.SCAFFOLD_SENSITIVE = real_ids


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
