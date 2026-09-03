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




def test_null_check_separates_drift_from_weak_power():
    """null 复现的核心分界: "差异出现了"是失败, "样本不够"只是警告 —— 不能混。
    第107轮我把 16 单元(MDE 0.46)的 null 当成"效应不存在", 脚本不许重犯。"""
    good = lambda p: _canon_for(p)
    # 两模型真等价: null 成立, 但 16 单元只能排除 >=46% -> WARN 而非静默 PASS
    problems, warnings = rf.check_model_null(good, good, n=4, verbose=False)
    assert problems == [], "两侧等价不该报失败"
    assert len(warnings) == 1 and "界更松" in warnings[0]
    assert "46%" in warnings[0] and "81 个单元" in warnings[0], \
        "警告必须给出这次的界与补齐所需样本量, 否则读者无法判断可信度"
    # 一侧明显更差: null 失效 -> FAIL, 且不许同时报 WARN(原因要单一明确)
    problems, warnings = rf.check_model_null(good, lambda p: "错", n=4, verbose=False)
    assert len(problems) == 1 and "null 已失效" in problems[0]
    assert warnings == [], "已判失效就不该再报界的警告"
    assert "不要直接改记录" in problems[0]
    # 样本降到无信息: 必须说"不能算复现", 而不是通过
    problems, warnings = rf.check_model_null(good, good, n=1, verbose=False)
    assert problems == [] and len(warnings) == 1
    assert "无信息" in warnings[0] and "不能算复现" in warnings[0]


def test_main_exit_codes_and_null_wiring():
    """warnings 不改变退出码(界松不是失败), problems 才改。null 检查仅在给出
    第二个模型时才跑 —— 否则单模型用户会看到一个凭空的 null 结论。"""
    good = lambda p: _canon_for(p)
    assert rf.main(strict_only) == 0                      # 只跑正向发现
    assert rf.main(strict_only, call_b=good, n=4) == 0    # 加 null: WARN 但仍 0
    assert rf.main(strict_only, call_b=lambda p: "错", n=4) == 1, \
        "null 失效必须让退出码非零"
    # 正向发现失败时也要非零, 且与 null 无关
    assert rf.main(good) == 1




def test_pass_line_names_which_checks_reproduced():
    """PASS 文案必须点名复现了哪几项 —— 一句光秃秃的"复现成功"无法区分
    "两项都过"与"只跑了一项", 而后者常因忘传第二个模型发生。"""
    import contextlib
    import io

    good = lambda p: _canon_for(p)

    def run(*a, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rf.main(*a, **kw)
        return rc, buf.getvalue()

    rc, out = run(strict_only)
    assert rc == 0 and "PASS  脚手架效应 复现成功" in out, out[-200:]
    assert "模型 null" not in out, "未传第二个模型时不该出现 null 结论"

    rc, out = run(strict_only, call_b=good, n=4)
    pass_line = [l for l in out.splitlines() if l.startswith("PASS")][0]
    assert "脚手架效应" in pass_line and "模型 null" in pass_line, pass_line
    assert "界更松" in pass_line, "界松这件事必须出现在 PASS 行本身, 不能只在 WARN 里"
    assert "不构成完整复现" in out

    # null 失效时 PASS 行不得出现
    rc, out = run(strict_only, call_b=lambda p: "错", n=4)
    assert rc == 1 and "PASS" not in out and "FAIL" in out


def test_null_bound_equal_to_record_is_full_reproduction():
    """界恰好等于记录值(0.10)时算完整复现, 不该报"更松" —— 边界要包含。"""
    real = rf.pb.report
    try:
        for bound, want_warn in ((0.10, False), (0.10 + 1e-9, True)):
            rf.pb.report = lambda *a, _b=bound, **k: {
                "pairs": [{"a": "a", "b": "b", "mean_diff": 0.0,
                           "p_mcnemar_holm": 1.0, "concentration": None,
                           "diff_ci": (0.0, 0.0),
                           "interpretation": {"verdict": "null", "rules_out": _b,
                                              "n_units": 80, "text": "(合成)"}}],
                "saturation": {"informative": 4, "n_tasks": 4}, "text": "(合成)"}
            _, warnings = rf.check_model_null(lambda p: "x", lambda p: "x",
                                              n=1, verbose=False)
            assert bool(warnings) == want_warn, \
                f"界={bound} 期望警告={want_warn}, 实得 {warnings}"
    finally:
        rf.pb.report = real


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
