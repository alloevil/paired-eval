#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复现检查脚本自身的测试: 用假 call 驱动 check_scaffold_effect 的四条判据。

为什么要测它: 这个脚本是"结论是否漂移"的唯一警报器。若它的判据写反(例如把
方向反转当成通过、把饱和当成成功), 漂移会静默通过 —— 与"门禁必须自测"同源(docs/lessons.md#gates-self-test)。
不需要真实模型: 注入的 call 是纯函数, 因此可以进快速套件。
"""
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 项目根: 让 `python3 tests/x.py` 直接可跑
from paired_eval import reproduce_findings as rf
from paired_eval import paired_bench as pb

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
    曾把 16 单元(MDE 0.46)的 null 当成"效应不存在"(docs/corrections.md#over-claimed-null), 脚本不许重犯。"""
    good = lambda p: _canon_for(p)
    # 两模型完全等价 -> 零不一致对。地板基数修正(docs/lessons.md#floor-basis)后这里的诊断变准了: 病因不是"界更松",
    # 而是"检验无力" —— McNemar 一个不一致对都没有, p 恒为 1.0, 与单元数无关。
    problems, warnings = rf.check_model_null(good, good, n=4, verbose=False)
    assert problems == [], "两侧等价不该报失败"
    assert len(warnings) == 1 and "不构成复现" in warnings[0]
    assert "检验无力" in warnings[0] and "不一致对只有 1 个" in warnings[0], \
        f"警告须复用 interpret 的准确诊断, 而非自己重推病因: {warnings[0]}"
    assert "80 单元" in warnings[0], "仍须给出记录值供对照"
    # 一侧明显更差: null 失效 -> FAIL, 且不许同时报 WARN(原因要单一明确)
    problems, warnings = rf.check_model_null(good, lambda p: "错", n=4, verbose=False)
    assert len(problems) == 1 and "null 已失效" in problems[0]
    assert warnings == [], "已判失效就不该再报界的警告"
    assert "不要直接改记录" in problems[0]
    # 样本降到最低: 同样只警告, 不许通过
    problems, warnings = rf.check_model_null(good, good, n=1, verbose=False)
    assert problems == [] and len(warnings) == 1
    assert "不构成复现" in warnings[0]


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




def _fixed_grader(single, checked):
    """替换 _grade_derivation: 按是否自检返回固定分数, 不触碰模型或 judge。"""
    return lambda case, call, judge, use_check: checked if use_check else single


def test_derivation_check_five_criteria():
    """推算增量复现的五条判据必须各指其病 —— 报错原因错了, 维护者会修错东西。
    这条结论推翻了第113/114轮的"纯替代", 直接改了实践建议, 故最该被监控。"""
    real = rf._grade_derivation
    try:
        # 1 增量正常: 完整复现
        rf._grade_derivation = _fixed_grader(0.79, 0.92)
        assert rf.check_derivation_increment(None, None, n=6, verbose=False) == ([], [])
        # 2a 两侧完全相同(零非零差值对): 地板基数修正(docs/lessons.md#floor-basis)后归为功效问题而非漂移 ——
        # 检验没有任何区分材料, 说"在足够样本下仍无法确认"是错的(样本并不足够)。
        rf._grade_derivation = _fixed_grader(0.79, 0.79)
        p, w = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert p == [] and len(w) == 1, (p, w)
        assert "检验无力" in w[0] and "非零差值对" in w[0], w
        # 2b 真正的"增量已衰减": 每对都有非零差值, 但幅度小于阈值 -> 才算漂移
        alt = [0.79 + 0.02, 0.79 + 0.01]
        seq = {"i": 0}

        def small_gain(case, call, judge, use_check):
            if not use_check:
                return 0.79
            v = alt[seq["i"] % len(alt)]
            seq["i"] += 1
            return v

        rf._grade_derivation = small_gain
        p, w = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert len(p) == 1 and w == [], (p, w)
        assert "已衰减" in p[0] and "+0.183" in p[0], p
        # 3 方向反转: 单独报出, 且明示不要当失败处理
        rf._grade_derivation = _fixed_grader(0.90, 0.50)
        p, w = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert len(p) == 1 and "方向反转" in p[0] and "重大发现" in p[0]
        assert w == [], "已判方向反转就不该再报其他"
        # 4 参照格触顶: 处方必须是换题, 而非放宽阈值(第113~115轮的教训)
        rf._grade_derivation = _fixed_grader(1.0, 1.0)
        p, w = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert len(p) == 1 and "触顶" in p[0] and "换题, 不是放宽阈值" in p[0]
        # 5 样本不足: 只警告, 且给出所需轮数 —— 不许把"没测出"说成"没有"
        rf._grade_derivation = _fixed_grader(0.79, 0.92)
        p, w = rf.check_derivation_increment(None, None, n=2, verbose=False)
        assert p == [] and len(w) == 1 and "n>=6" in w[0], (p, w)
        # 边界: 恰好 18 单元(n=6 x 3 题)不该报样本不足
        p, w = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert w == [], "18 单元恰好达标, 不该警告"
    finally:
        rf._grade_derivation = real


def test_main_wires_judge_only_when_given():
    """未传 judge 时不得跑推算检查 —— 否则单模型无 judge 的用户会看到凭空结论。"""
    import contextlib
    import io

    real = rf._grade_derivation
    try:
        rf._grade_derivation = _fixed_grader(0.79, 0.92)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rf.main(strict_only)
        assert rc == 0 and "推算增量" not in buf.getvalue()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rf.main(strict_only, judge=object(), n_deriv=6)
        out = buf.getvalue()
        assert rc == 0 and "=== 3 推算增量" in out
        pass_line = [l for l in out.splitlines() if l.startswith("PASS")][0]
        assert "推算增量" in pass_line and "样本不足" not in pass_line, pass_line
        # 样本不足时 PASS 行须标注, 与 null 检查同规矩
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rf.main(strict_only, judge=object(), n_deriv=2)
        pass_line = [l for l in buf.getvalue().splitlines() if l.startswith("PASS")][0]
        assert "样本不足" in pass_line, pass_line
    finally:
        rf._grade_derivation = real




def test_derivation_thresholds_are_inclusive():
    """三处阈值恰好相等时都算通过 —— 否则正常噪声会在边界上随机报警。
    变异测试抓到这三处: 触顶阈 / 增量阈 / 单元数下界的 == 情形全未被覆盖。"""
    E = rf.EXPECTED
    real = rf._grade_derivation
    real_inc = E["derivation_increment_min"]
    try:
        # 触顶阈: 参照格恰好 0.95 不算触顶(此时报的是增量不足, 病因不同)
        rf._grade_derivation = _fixed_grader(E["derivation_ceiling_max"], 1.0)
        p, _ = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert len(p) == 1 and "触顶" not in p[0], p
        rf._grade_derivation = _fixed_grader(E["derivation_ceiling_max"] + 0.001, 1.0)
        p, _ = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert "触顶" in p[0], p
        # 增量阈: 恰好等于阈值通过, 差一点点则失败。
        # 必须用二进制精确的值: 初版用 0.88-0.80, 它其实是 0.07999999999999996(< 0.08), 在
        # 3.10 的朴素 sum() 下 18 次舍入误差恰好把均值推回 0.08 而"通过", 3.12 起 sum() 改用
        # Neumaier 补偿求和, 结果更准, 刀锋翻转 —— CI 第一次运行就在 3.12 上抓到。
        E["derivation_increment_min"] = 0.125                    # 二进制精确
        base = 0.5
        assert (0.625 - base) == 0.125, "构造值必须精确, 否则又是刀锋"
        rf._grade_derivation = _fixed_grader(base, 0.625)
        assert rf.check_derivation_increment(None, None, n=6, verbose=False) == ([], [])
        rf._grade_derivation = _fixed_grader(base, 0.625 - 1e-3)
        p, _ = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert len(p) == 1 and "已衰减" in p[0], p
        E["derivation_increment_min"] = real_inc
        # 单元数下界: 恰好 18 通过; 15(n=5)警告并给出所需轮数
        rf._grade_derivation = _fixed_grader(0.79, 0.92)
        assert rf.check_derivation_increment(None, None, n=6, verbose=False)[1] == []
        p, w = rf.check_derivation_increment(None, None, n=5, verbose=False)
        assert p == [] and "单元数 15 < 18" in w[0] and "n>=6" in w[0], (p, w)
    finally:
        rf._grade_derivation = real
        E["derivation_increment_min"] = real_inc      # 断言中途失败时也要还原, 否则污染后续测试




def test_ceiling_threshold_exact_equality():
    """真正测"参照格恰好等于触顶阈"需要二进制可精确表示的值 —— 用 0.95 时
    sum([0.95]*18)/18 = 0.9499999999999997, 相等分支从未被走到, 变异测试抓到了这点。
    契约: 恰好等于阈值不算触顶(与其他阈值一致, 边界包含)。"""
    real_g, real_c = rf._grade_derivation, rf.EXPECTED["derivation_ceiling_max"]
    try:
        rf.EXPECTED["derivation_ceiling_max"] = 0.5      # 0.5 在二进制中精确
        assert sum([0.5] * 18) / 18 == 0.5, "该值必须精确, 否则又测不到相等"
        rf._grade_derivation = _fixed_grader(0.5, 0.75)
        p, _ = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert not any("触顶" in x for x in p), f"ref 恰好等于阈值不该判触顶: {p}"
        # 超出一个最小浮点步长即判触顶
        import math
        rf._grade_derivation = _fixed_grader(math.nextafter(0.5, 1.0), 0.75)
        p, _ = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert any("触顶" in x for x in p), f"ref 略超阈值必须判触顶: {p}"
    finally:
        rf._grade_derivation = real_g
        rf.EXPECTED["derivation_ceiling_max"] = real_c




def test_derivation_ci_straddling_zero_fails():
    """CI 下界为负 = 证据没排除"增量为零", 必须判失败(而非警告): 此时非零差值对
    充足, 检验有力, 只是结论不成立 —— 这正是"漂移"与"功效不足"的分界线。

    附一条结构性观察: 下界"恰好等于 0"在本函数里不可达 —— 要让 bootstrap
    下界落在 0, 需 <=3 个非零差值对(否则全零重采样的概率低于 2.5%), 而那时"检验无力"
    分支(需 >=6 个非零对才放行)已先拦下。故 `<= 0` 与 `< 0` 在此结构下同结果。"""
    real = rf._grade_derivation
    try:
        # 多数对小幅正、少数对大幅负 -> 均值正但 CI 跨 0; 18 对全非零, 检验有力
        seq = [0.10, 0.10, 0.10, 0.10, -0.30, 0.10]
        st = {"i": 0}

        def mixed(case, call, judge, use_check):
            if not use_check:
                return 0.5
            v = 0.5 + seq[st["i"] % len(seq)]
            st["i"] += 1
            return v

        rf._grade_derivation = mixed
        p, w = rf.check_derivation_increment(None, None, n=6, verbose=False)
        assert w == [], f"18 对全非零, 不该报功效问题: {w}"
        assert len(p) == 1 and "CI 下界" in p[0], p
        assert "+0.086" in p[0], "须点出记录值供对照"
        assert "-" in p[0].split("CI 下界")[1][:8], f"下界应为负: {p[0]}"
    finally:
        rf._grade_derivation = real


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
