# -*- coding: utf-8 -*-
"""paired_bench 离线测试:mock 模型,零网络。运行: python3 test_paired_bench.py"""
import paired_bench as pb


def _canon(prompt):
    """按 instruction 匹配任务,返回其 canonical 正确输出。"""
    for t in pb.ALL_TASKS:
        if t["instruction"] in prompt:
            return t["canonical"]
    raise AssertionError(f"未知prompt: {prompt[:40]}")


def test_all_canonicals_pass_their_checks():
    """自洽性: 每个任务的 canonical 样例必须通过它自己的判定器 —— 判定器与样例互为校验。"""
    for t in pb.ALL_TASKS:
        assert t["check"](t["canonical"]), f"canonical 过不了自家判定器: {t['id']}"


def test_ids_unique_and_shape():
    ids = [t["id"] for t in pb.ALL_TASKS]
    assert len(ids) == len(set(ids)), "任务 id 必须唯一"
    assert len(pb.ALL_TASKS) == 28


def test_run_paired_flow():
    wrong_ids = {"if-upper", "ch-rev6", "qa-2p10"}

    def model_a(prompt):
        return _canon(prompt)

    def model_b(prompt):
        for t in pb.ALL_TASKS:
            if t["instruction"] in prompt:
                return "错误输出!!!" if t["id"] in wrong_ids else t["canonical"]
        return None

    out = pb.run_paired(model_a, model_b)
    assert not out["dropped"]
    c = out["compare"]
    assert c["n"] == 28 and c["wins"] == 3 and c["losses"] == 0 and c["ties"] == 25
    assert abs(c["mean_diff"] - 3 / 28) < 1e-9


def test_paired_drop_and_crash_tolerance():
    def model_a(prompt):
        if "if-json" in str([t["id"] for t in pb.ALL_TASKS if t["instruction"] in prompt]):
            return None            # 拒答 -> 成对丢弃
        return _canon(prompt)

    def model_b(prompt):
        for t in pb.ALL_TASKS:
            if t["instruction"] in prompt:
                if t["id"] == "if-3lines":
                    return "not json at all"   # 判定器抛异常也只算 0 分,不崩
                return t["canonical"]
        return None

    out = pb.run_paired(model_a, model_b)
    assert out["dropped"] == ["if-json"]
    assert out["compare"]["n"] == 27
    row = next(r for r in out["rows"] if r["id"] == "if-3lines")
    assert row["b"] == 0.0 and row["a"] == 1.0


def test_run_repeated_separates_capability_from_reliability():
    tasks = [t for t in pb.ALL_TASKS if t["id"] in ("if-upper", "if-pi8")]
    calls = {"n": 0}

    def flaky(prompt):   # if-upper 恒对(能力); if-pi8 隔次对错(可靠性问题)
        for t in tasks:
            if t["instruction"] in prompt:
                if t["id"] == "if-upper":
                    return t["canonical"]
                calls["n"] += 1
                return t["canonical"] if calls["n"] % 2 == 1 else "错误"
        raise AssertionError("未知prompt")

    rep = {r["id"]: r for r in pb.run_repeated(flaky, tasks=tasks, n=8, k=3)}
    stable = rep["if-upper"]
    assert stable["successes"] == 8 and stable["pass_at_1"] == 1.0 and stable["pass_hat_k"] == 1.0
    coin = rep["if-pi8"]
    assert coin["successes"] == 4 and coin["pass_at_1"] == 0.5
    assert abs(coin["pass_hat_k"] - 4 / 56) < 1e-12, "C(4,3)/C(8,3): 半对的可靠性≈0.07"
    # 拒答按失败计
    rep2 = pb.run_repeated(lambda p: None, tasks=tasks[:1], n=3, k=2)
    assert rep2[0]["successes"] == 0 and rep2[0]["pass_hat_k"] == 0.0



def test_make_model_adapter():
    calls, naps = [], []

    def flaky(prompt):        # 第一次炸,第二次成功
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return "答案: 42"

    m = pb.make_model(flaky, tries=2, sleep=naps.append)
    assert m("q") == "答案: 42" and naps == [2.0], "瞬时故障重试后成功"
    dead = pb.make_model(lambda p: (_ for _ in ()).throw(RuntimeError("down")),
                         tries=2, sleep=lambda _: None)
    assert dead("q") is None, "持续故障返回None走成对丢弃,不崩批"
    # 全灭防静默: 两侧全None时 run_paired 必须报错而非返回空结果
    try:
        pb.run_paired(dead, dead, tasks=pb.ALL_TASKS[:2])
        assert False, "全部丢弃应报错"
    except ValueError:
        pass



def test_reliability_matrix():
    mk_report = lambda pairs: [{"id": i, "n": 8, "successes": int(p * 8),
                                "pass_at_1": p, "pass_hat_k": 0.0, "runs": []}
                               for i, p in pairs]
    reports = {
        "smol":    mk_report([("rev6", 1.0), ("third", 0.5)]),
        "default": mk_report([("rev6", 0.0), ("third", 1.0)]),
        "slow":    mk_report([("rev6", 0.875), ("third", 1.0)]),
    }
    rows = pb.reliability_matrix(reports)
    r6 = next(r for r in rows if r["id"] == "rev6")
    assert r6["smol"] == 1.0 and r6["default"] == 0.0 and r6["slow"] == 0.875
    assert abs(r6["spread"] - 1.0) < 1e-9 and r6["divergent"]
    th = next(r for r in rows if r["id"] == "third")
    assert abs(th["spread"] - 0.5) < 1e-9 and th["divergent"]
    # 阈值可调: 提高到0.6后 third 不再算分歧
    rows2 = pb.reliability_matrix(reports, divergence=0.6)
    assert not next(r for r in rows2 if r["id"] == "third")["divergent"]
    # 任务集不一致必须拒绝
    bad = dict(reports, slow=mk_report([("rev6", 1.0)]))
    try:
        pb.reliability_matrix(bad)
        assert False, "id序列不同应报错"
    except ValueError:
        pass



def test_audit_trails():
    tasks = [t for t in pb.ALL_TASKS if t["id"] == "if-upper"]

    def model_a(prompt):
        return "HELLO WORLD"

    def model_b(prompt):
        return "x" * 500   # 超长错误输出必须被截断留痕

    out = pb.run_paired(model_a, model_b, tasks=tasks)
    row = out["rows"][0]
    assert row["resp_a"] == "HELLO WORLD" and row["b"] == 0.0
    assert row["resp_b"] == "x" * 120, "留痕按 _TRAIL_CAP 截断"
    rep = pb.run_repeated(lambda p: "HELLO WORLD", tasks=tasks, n=3)
    assert rep[0]["responses"] == ["HELLO WORLD"] * 3
    rep2 = pb.run_repeated(lambda p: None, tasks=tasks, n=2)
    assert rep2[0]["responses"] == [None, None], "拒答留痕为 None,与失败输出可区分"



def test_wrong_probes_rejected():
    """outcome validity 的另一半: canonical 必须被接受(已有测试),
    通用错误输出必须被拒绝 —— 来者不拒的判定器会让所有系统并列满分(ABC do-nothing 类缺陷)。
    判定器抛异常按拒绝计(与 run_paired 的 score 语义一致)。"""
    probes = ["", "这是一段不相关的回答内容", "答案: 9999999999"]
    leaky = []
    for t in pb.ALL_TASKS:
        for p in probes:
            try:
                accepted = bool(t["check"](p))
            except Exception:
                accepted = False
            if accepted:
                leaky.append((t["id"], p[:20]))
    assert not leaky, f"判定器接受了明显错误的输出: {leaky}"



def test_measurement_window_guard():
    """可靠性跨会话漂移(实测 0/8 -> 8/8), 故不同时间窗的报告不许并排。"""
    tasks = [t for t in pb.ALL_TASKS if t["id"] == "if-upper"]
    fresh = pb.run_repeated(lambda p: "HELLO WORLD", tasks=tasks, n=2)
    assert isinstance(fresh[0]["measured_at"], float), "报告必须带测量时戳"
    same_window = pb.reliability_matrix({"a": fresh, "b": fresh})
    assert len(same_window) == 1 and not same_window[0]["divergent"]
    stale = [dict(r, measured_at=r["measured_at"] - 7200) for r in fresh]
    try:
        pb.reliability_matrix({"a": fresh, "b": stale})
        assert False, "跨2小时的报告并排应报错"
    except ValueError as e:
        assert "漂移" in str(e)
    assert pb.reliability_matrix({"a": fresh, "b": stale}, max_span_s=None,
                                 require_interleaved=False), "两道守卫均可显式关闭"
    # 合成报告(无时戳)不受影响: 向后兼容
    synth = [{"id": "if-upper", "n": 8, "successes": 8, "pass_at_1": 1.0,
              "pass_hat_k": 1.0, "runs": []}]
    assert pb.reliability_matrix({"a": synth, "b": synth})


def test_sequential_measurement_rejected():
    """逐系统分别 run_repeated 再并排 = 顺序测量(错误路径), 必须被拦下 ——
    本仓库正是这样得出过假的"非单调洞"结论。"""
    tasks = [t for t in pb.ALL_TASKS if t["id"] == "if-upper"]
    good = lambda p: "HELLO WORLD"
    bad = lambda p: "hello world"
    seq = {"a": pb.run_repeated(good, tasks=tasks, n=2),
           "b": pb.run_repeated(bad, tasks=tasks, n=2)}
    try:
        pb.reliability_matrix(seq)
        assert False, "顺序测量应被拒绝"
    except ValueError as e:
        assert "顺序测量" in str(e)
    assert pb.reliability_matrix(seq, require_interleaved=False), "可显式承担风险"
    # 交错路径同题共享时戳,顺利通过
    inter = pb.run_interleaved({"a": good, "b": bad}, tasks=tasks, n=2)["reports"]
    rows = pb.reliability_matrix(inter)
    assert rows[0]["a"] == 1.0 and rows[0]["b"] == 0.0 and rows[0]["divergent"]



def test_discordant_readout():
    """逐轮不一致对 + McNemar: 逐题均值检验之外的第二个读数(保留逐轮结构)。"""
    tasks = [t for t in pb.ALL_TASKS if t["id"] in ("if-upper", "if-pi8")]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    good = lambda p: canon[[k for k in canon if k in p][0]]
    bad = lambda p: "错"
    out = pb.run_paired_repeated(good, bad, tasks=tasks, n=3)
    d = out["discordant"]
    assert d["a_only"] == 6 and d["b_only"] == 0, "2题×3轮全为A独对"
    assert abs(d["mcnemar_p"] - 2 * (1 / 64)) < 1e-12, "2*C(6,0)/2^6"
    # 两侧同强: 无不一致对 -> p=1
    out2 = pb.run_paired_repeated(good, good, tasks=tasks, n=2)
    assert out2["discordant"] == {"a_only": 0, "b_only": 0, "mcnemar_p": 1.0}



def test_pairwise_compare_multiplicity():
    """N系统两两比较: 双读数 + 各家族独立 Holm 校正; 守卫复用。"""
    tasks = pb.ALL_TASKS[:6]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    good = lambda p: canon[[k for k in canon if k in p][0]]
    bad = lambda p: "错误输出"
    reps = pb.run_interleaved({"g1": good, "g2": good, "b1": bad}, tasks=tasks, n=2)["reports"]
    rows = pb.pairwise_compare(reps)
    assert len(rows) == 3, "3系统 -> 3对"
    assert [abs(r["mean_diff"]) for r in rows] == sorted(
        (abs(r["mean_diff"]) for r in rows), reverse=True), "按效应量降序"
    by = {frozenset((r["a"], r["b"])): r for r in rows}
    gg = by[frozenset(("g1", "g2"))]
    assert gg["mean_diff"] == 0.0 and gg["a_only"] == gg["b_only"] == 0
    assert gg["mcnemar_p"] == 1.0 and gg["p_mcnemar_holm"] == 1.0
    gb = by[frozenset(("b1", "g1"))]
    assert abs(gb["mean_diff"]) == 1.0 and gb["a_only"] + gb["b_only"] == 12  # 6题×2轮
    assert gb["mcnemar_p"] < 1e-3 and gb["p_mcnemar_holm"] >= gb["mcnemar_p"], "校正只会变大"
    assert gb["p_mcnemar_holm"] < 0.05, "真实强效应经校正仍显著"
    # 守卫复用: 顺序测量的报告应被拒
    seq = {"a": pb.run_repeated(good, tasks=tasks[:1], n=2),
           "b": pb.run_repeated(bad, tasks=tasks[:1], n=2)}
    try:
        pb.pairwise_compare(seq)
        assert False, "顺序测量应被拒绝"
    except ValueError:
        pass



def test_saturation_diagnostic():
    """饱和题(全系统恒过/恒败)不携带区分信息, 必须与 informative 分开计数 ——
    否则"我有28道题"会掩盖"其中26道毫无区分力"。"""
    tasks = pb.ALL_TASKS[:4]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    easy_ids = {tasks[0]["id"], tasks[1]["id"]}
    hard_id = tasks[2]["id"]
    split_id = tasks[3]["id"]

    def make(win_split):
        def model(prompt):
            t = next(t for t in tasks if t["instruction"] in prompt)
            if t["id"] in easy_ids:
                return canon[t["instruction"]]          # 两系统恒过
            if t["id"] == hard_id:
                return "错"                              # 两系统恒败
            return canon[t["instruction"]] if win_split else "错"   # 唯一有信息的题
        return model

    reps = pb.run_interleaved({"a": make(True), "b": make(False)}, tasks=tasks, n=2)["reports"]
    s = pb.saturation(reps)
    assert s["n_tasks"] == 4 and s["saturated_pass"] == 2 and s["saturated_fail"] == 1
    assert s["informative"] == 1 and abs(s["informative_rate"] - 0.25) < 1e-9
    assert s["ids"]["informative"] == [split_id]
    assert set(s["ids"]["saturated_pass"]) == easy_ids
    assert s["ids"]["saturated_fail"] == [hard_id]
    # 守卫复用
    seq = {"a": pb.run_repeated(make(True), tasks=tasks[:1], n=1),
           "b": pb.run_repeated(make(False), tasks=tasks[:1], n=1)}
    try:
        pb.saturation(seq)
        assert False, "顺序测量应被拒绝"
    except ValueError:
        pass



def test_screen_tasks():
    """入库前筛选: 只保留有区分力的题; 坏候选(canonical不自洽/id冲突/字段缺失)先被拦。"""
    cands = [
        {"id": "cand-easy", "instruction": "输出 OK", "check": lambda r: r.strip() == "OK",
         "canonical": "OK"},
        {"id": "cand-hard", "instruction": "输出 IMPOSSIBLE",
         "check": lambda r: r.strip() == "IMPOSSIBLE", "canonical": "IMPOSSIBLE"},
        {"id": "cand-split", "instruction": "输出 SPLIT", "check": lambda r: r.strip() == "SPLIT",
         "canonical": "SPLIT"},
    ]
    good = lambda p: {"输出 OK": "OK", "输出 IMPOSSIBLE": "no", "输出 SPLIT": "SPLIT"}[
        next(c["instruction"] for c in cands if c["instruction"] in p)]
    weak = lambda p: {"输出 OK": "OK", "输出 IMPOSSIBLE": "no", "输出 SPLIT": "nope"}[
        next(c["instruction"] for c in cands if c["instruction"] in p)]
    out = pb.screen_tasks(cands, {"a": good, "b": weak}, n=2, confirm_n=4)
    assert [t["id"] for t in out["kept"]] == ["cand-split"], "稳定差异经复核仍留"
    assert [t["id"] for t in out["flagged"]] == ["cand-split"]
    assert out["dropped_on_confirm"] == [] and out["confirm_saturation"]["informative"] == 1
    assert [t["id"] for t in out["saturated_pass"]] == ["cand-easy"]
    assert [t["id"] for t in out["saturated_fail"]] == ["cand-hard"]
    assert out["saturation"]["informative"] == 1
    # 噪声信号: 筛选阶段偶然分歧, 复核阶段消失 -> 必须被剔除(实测发生过)
    flip = {"i": 0}

    def noisy(prompt):
        if "SPLIT" not in prompt:
            return good(prompt)
        flip["i"] += 1
        return "SPLIT" if flip["i"] == 1 else "nope"   # 只在第一次答对

    out2 = pb.screen_tasks(cands, {"a": noisy, "b": weak}, n=2, confirm_n=4)
    assert [t["id"] for t in out2["flagged"]] == ["cand-split"], "低n筛选把噪声标成有信息"
    assert out2["kept"] == [] and [t["id"] for t in out2["dropped_on_confirm"]] == ["cand-split"], \
        "复核阶段必须剔除噪声信号"
    # 可跳过复核
    out3 = pb.screen_tasks(cands, {"a": good, "b": weak}, n=2, confirm_n=None)
    assert [t["id"] for t in out3["kept"]] == ["cand-split"] and out3["confirm_saturation"] is None
    # 坏候选拦截
    bad_cases = [
        [{"id": "x", "instruction": "i", "check": lambda r: False, "canonical": "c"}],
        [{"id": pb.ALL_TASKS[0]["id"], "instruction": "i",
          "check": lambda r: True, "canonical": "c"}],
        [{"id": "y", "instruction": "i"}],
    ]
    for bad in bad_cases:
        try:
            pb.screen_tasks(bad, {"a": good, "b": weak}, n=1)
            assert False, f"坏候选应被拦: {bad[0].get('id')}"
        except ValueError:
            pass



def test_paired_repeated_survives_drift():
    """交错配对的核心性质: 全局漂移(两侧同时变差)不产生假差异;
    真实单侧差异照常检出; 拒答按轮成对丢弃,配对不破。"""
    tasks = [t for t in pb.ALL_TASKS if t["id"] in ("if-upper", "if-pi8")]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    state = {"i": 0}

    def drifting(prompt):   # 前半窗全对, 后半窗全错 —— 对两侧同等作用
        state["i"] += 1
        return canon[[k for k in canon if k in prompt][0]] if state["i"] <= 16 else "错"

    out = pb.run_paired_repeated(drifting, drifting, tasks=tasks, n=8)
    c = out["compare"]
    assert c["wins"] == 0 and c["losses"] == 0, f"漂移不得制造假差异: {c}"
    assert all(r["a_pass_at_1"] == r["b_pass_at_1"] for r in out["rows"])
    # 真实单侧差异仍可检出
    good = lambda p: canon[[k for k in canon if k in p][0]]
    bad = lambda p: "错误输出"
    out2 = pb.run_paired_repeated(good, bad, tasks=tasks, n=4)
    assert out2["compare"]["wins"] == 2 and out2["compare"]["losses"] == 0
    assert all(r["a_pass_hat_k"] == 1.0 and r["b_pass_hat_k"] == 0.0 for r in out2["rows"])
    # 拒答: 该轮两侧同弃, n 降为有效轮数
    flaky = {"i": 0}

    def sometimes_none(p):
        flaky["i"] += 1
        return None if flaky["i"] % 2 == 0 else good(p)

    out3 = pb.run_paired_repeated(good, sometimes_none, tasks=tasks[:1], n=4)
    row = out3["rows"][0]
    assert row["dropped_reps"] + row["n"] == 4 and row["n"] == len(row["per_rep"])
    # 全灭仍响亮
    try:
        pb.run_paired_repeated(lambda p: None, good, tasks=tasks[:1], n=2)
        assert False, "全部丢弃应报错"
    except ValueError:
        pass



def test_call_order_alternates_and_cancels_bias():
    """先叫谁必须逐题/逐轮交替: 否则"第一个被调用者占优"这类顺序效应
    会被读成系统差异(与 judge position bias 同源)。"""
    tasks = [t for t in pb.ALL_TASKS if t["id"] in ("if-upper", "if-pi8")]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    # 顺序偏置模型: 每对里第一个被调用的答对,第二个答错
    slot = {"i": 0}

    def biased(prompt):
        slot["i"] += 1
        first = slot["i"] % 2 == 1
        return canon[[k for k in canon if k in prompt][0]] if first else "错"

    out = pb.run_paired(biased, biased, tasks=tasks)
    assert [r["a_first"] for r in out["rows"]] == [True, False], "逐题交替"
    c = out["compare"]
    assert c["wins"] == c["losses"] == 1 and c["mean_diff"] == 0.0, \
        f"顺序偏置必须两侧对消,不得产出假赢家: {c}"
    slot["i"] = 0
    out2 = pb.run_paired_repeated(biased, biased, tasks=tasks[:1], n=4)
    row = out2["rows"][0]
    assert row["orders"] == [True, False, True, False], "逐轮交替(ABBA)"
    assert row["a_successes"] == row["b_successes"] == 2, "四轮下顺序优势各得其半"
    assert out2["compare"]["mean_diff"] == 0.0



def test_run_interleaved_rotates_and_feeds_matrix():
    """N系统交错: 顺序逐轮轮转, 首位优势在 n=系统数 的整数倍下完全对消;
    输出直接喂 reliability_matrix(同窗同轮次, 无跨窗混淆)。"""
    tasks = [t for t in pb.ALL_TASKS if t["id"] in ("if-upper", "if-pi8")]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    slot = {"i": 0}

    def first_wins(prompt):   # 每轮三次调用中第一次答对,其余答错
        slot["i"] += 1
        return canon[[k for k in canon if k in prompt][0]] if slot["i"] % 3 == 1 else "错"

    models = {"m1": first_wins, "m2": first_wins, "m3": first_wins}
    out = pb.run_interleaved(models, tasks=tasks, n=3)
    reps = out["reports"]
    assert set(reps) == {"m1", "m2", "m3"} and not out["dropped"]
    assert reps["m1"][0]["orders"] == [["m1", "m2", "m3"], ["m2", "m3", "m1"],
                                      ["m3", "m1", "m2"]], "顺序逐轮轮转"
    for nm in models:
        assert reps[nm][0]["successes"] == 1, f"{nm} 应各得一次首位优势"
    rows = pb.reliability_matrix(reps)
    assert len(rows) == 2 and not any(r["divergent"] for r in rows), \
        "纯顺序效应不得在矩阵里显示为系统差异"
    # 真实差异仍可见
    good = lambda p: canon[[k for k in canon if k in p][0]]
    rows2 = pb.reliability_matrix(pb.run_interleaved(
        {"good": good, "bad": lambda p: "错", "also_good": good}, tasks=tasks, n=2)["reports"])
    assert all(r["good"] == 1.0 and r["bad"] == 0.0 and r["divergent"] for r in rows2)
    # 单系统拒绝; 全弃响亮
    for bad_call in (lambda: pb.run_interleaved({"only": good}, tasks=tasks),
                     lambda: pb.run_interleaved({"a": lambda p: None, "b": good}, tasks=tasks, n=2)):
        try:
            bad_call()
            assert False, "应报错"
        except ValueError:
            pass



def test_trajectory_gate_fixture():
    """门禁 fixture 的结构不变量(无需LLM): 篡改必须真的偏离资料, 忠实回答必须真的贴合资料。
    fixture 若不满足这些, 门禁跑出的 recall/fp_rate 就毫无意义。"""
    assert len(pb.TRAJECTORY_GATE) >= 1
    case = pb.TRAJECTORY_GATE[0]
    obs = case["observations"][0]["observation"]
    assert obs and case["faithful"] and len(case["planted"]) == 2
    assert {p["expect"] for p in case["planted"]} == {"distorted", "fabricated"}
    # 忠实回答的关键数字必须都出现在资料里
    for num in ("1100亿美元", "540亿美元", "850亿美元", "290亿美元"):
        assert num in obs and num in case["faithful"], f"忠实回答与资料应共有 {num}"
    dist = next(p["response"] for p in case["planted"] if p["expect"] == "distorted")
    # 篡改后的数字必须不在资料里(否则"歪曲"其实有据, fixture 无效)
    for num in ("2000亿美元", "900亿美元", "超过200%"):
        assert num in dist and num not in obs, f"篡改数字 {num} 不应出现在资料中"
    assert dist != case["faithful"]
    fab = next(p["response"] for p in case["planted"] if p["expect"] == "fabricated")
    assert fab.startswith(case["faithful"]) and len(fab) > len(case["faithful"]), \
        "编造变体应是在忠实回答后追加内容"
    for num in ("净利润", "员工总数"):
        assert num in fab and num not in obs, f"追加内容 {num} 不应在资料中"


def test_world_and_rubric_gate_fixtures():
    """两个门禁 fixture 的结构不变量(无需LLM)。fixture 悄悄失效的门禁比没门禁更危险:
    它会持续输出漂亮的 recall 而实际什么都没测。"""
    # WORLD_GATE: 必须同时含"带植入错误的文档"与"干净文档"(后者才能测误报)
    assert len(pb.WORLD_GATE) >= 2
    dirty = [c for c in pb.WORLD_GATE if c["planted"]]
    clean = [c for c in pb.WORLD_GATE if not c["planted"]]
    assert dirty and clean, "缺干净文档就只能测查全,发现不了来者皆错的判定器"
    for c in dirty:
        for p in c["planted"]:
            assert p["substr"] in c["text"], f"植入特征 {p['substr']} 必须真的出现在文档里"
            assert p.get("desc"), "植入错误必须写明真值,否则无法复验"
    for c in clean:
        for p in dirty[0]["planted"]:
            assert p["substr"] not in c["text"], \
                f"干净文档不得含植入特征 {p['substr']}(否则误报统计被污染)"
    # RUBRIC_GATE: criteria 合法, 且 fooling 必须不含 good 的具体信息
    g = pb.RUBRIC_GATE
    assert g["criteria"] and all(c["text"] and c["weight"] > 0 for c in g["criteria"])
    assert g["good"] != g["fooling"]
    for token in ("640", "4%"):
        assert token in g["good"] and token not in g["fooling"], \
            f"fooling 不得含具体数据 {token} —— 否则它不是空洞奉承,分离度失去意义"
    assert len(g["fooling"]) >= 20, "fooling 应是像样的长句(靠长度也骗不过判定器)"



def test_discordance_distribution():
    """不一致对的分布必须可见: 总数相同但集中在一题 vs 散布多题, 诊断完全不同,
    而 p 值对两者一视同仁。"""
    tasks = pb.ALL_TASKS[:4]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    good = lambda p: canon[[k for k in canon if k in p][0]]

    def only_first_task_bad(prompt):
        t = next(t for t in tasks if t["instruction"] in prompt)
        return "错" if t["id"] == tasks[0]["id"] else canon[t["instruction"]]

    def every_task_flaky(prompt):
        t = next(t for t in tasks if t["instruction"] in prompt)
        every_task_flaky.n[t["id"]] = every_task_flaky.n.get(t["id"], 0) + 1
        return canon[t["instruction"]] if every_task_flaky.n[t["id"]] % 2 == 0 else "错"
    every_task_flaky.n = {}

    # 集中型: 4题×2轮 = 全部不一致对都来自第1题
    conc = pb.pairwise_compare(pb.run_interleaved(
        {"good": good, "bad1": only_first_task_bad}, tasks=tasks, n=2)["reports"])[0]
    assert conc["a_only"] + conc["b_only"] == 2 and len(conc["by_task"]) == 1
    assert conc["concentration"] == 1.0, "单题贡献全部不一致对"
    assert conc["by_task"][0]["id"] == tasks[0]["id"]
    # 分散型: 每题各贡献一次
    spread = pb.pairwise_compare(pb.run_interleaved(
        {"good": good, "flaky": every_task_flaky}, tasks=tasks, n=2)["reports"])[0]
    assert len(spread["by_task"]) == 4, "四题各有不一致对"
    assert spread["concentration"] < 0.5, f"分散型集中度应低: {spread['concentration']}"
    # 无分歧时集中度为 None(不是0,"没有分布"与"分布均匀"是两回事)
    same = pb.pairwise_compare(pb.run_interleaved(
        {"g1": good, "g2": good}, tasks=tasks, n=2)["reports"])[0]
    assert same["by_task"] == [] and same["concentration"] is None


def test_refusal_attribution():
    """拒答归属必须可见: 一侧全拒(安全过滤/不可用)与双侧偶发是不同的系统事实,
    只报"丢弃N题"会把前者伪装成对称损耗。"""
    tasks = pb.ALL_TASKS[:3]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    good = lambda p: canon[[k for k in canon if k in p][0]]
    refuser = lambda p: None                     # 该侧全拒

    out = pb.run_paired(good, good, tasks=tasks)
    assert out["dropped"] == [] and out["dropped_detail"] == []
    try:                                          # 全拒时仍响亮报错
        pb.run_paired(good, refuser, tasks=tasks)
        assert False, "全部丢弃应报错"
    except ValueError:
        pass
    # 只在特定题上拒答: 归属到具体侧
    picky = lambda p: None if tasks[1]["instruction"] in p else good(p)
    out2 = pb.run_paired(good, picky, tasks=tasks)
    assert out2["dropped"] == [tasks[1]["id"]]
    assert out2["dropped_detail"] == [{"id": tasks[1]["id"], "sides": ["b"]}], \
        "必须指出是 b 侧拒答"
    # 重复配对: 逐题记录各侧拒答次数
    flaky = {"i": 0}

    def half_refuse(p):
        flaky["i"] += 1
        return None if flaky["i"] % 2 == 0 else good(p)

    rep = pb.run_paired_repeated(good, half_refuse, tasks=tasks[:1], n=4)
    row = rep["rows"][0]
    assert row["refusals"] == {"b": 2} and row["dropped_reps"] == 2, \
        f"拒答应归到 b 侧: {row['refusals']}"
    # N系统交错: 按系统名归属
    inter = pb.run_interleaved({"ok": good, "picky": half_refuse}, tasks=tasks[:1], n=4)
    r0 = inter["reports"]["ok"][0]
    assert r0["refusals"] and set(r0["refusals"]) <= {"ok", "picky"}
    assert "picky" in r0["refusals"], "拒答方必须被点名"


def test_batch_refusal_summary():
    """批次级拒答率必须直接可读: 埋在逐题字典里等于没有 ——
    一个系统拒答15%时, 它在剩下85%上的可比性本身就需要标注。
    整题被弃(其 row 不入表)的拒答同样必须计入, 否则总数偏低。"""
    tasks = pb.ALL_TASKS[:2]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    good = lambda p: canon[[k for k in canon if k in p][0]]
    # b 侧只在第2题拒答且全拒 -> 该题整体被弃, 其拒答不得漏计
    picky = lambda p: None if tasks[1]["instruction"] in p else good(p)
    out = pb.run_paired_repeated(good, picky, tasks=tasks, n=3)
    assert out["dropped"] == [tasks[1]["id"]], "第2题整体被弃"
    assert out["attempts_per_side"] == 6, "2题×3轮"
    assert out["refusals"] == {"a": 0, "b": 3}, \
        f"整题被弃的3次拒答必须计入: {out['refusals']}"
    assert out["refusal_rate"]["b"] == 0.5 and out["refusal_rate"]["a"] == 0.0
    # N系统同样
    inter = pb.run_interleaved({"ok": good, "picky": picky}, tasks=tasks, n=3)
    assert inter["attempts_per_system"] == 6
    assert inter["refusals"] == {"ok": 0, "picky": 3}
    assert inter["refusal_rate"]["picky"] == 0.5
    # 无拒答时为零而非None("零拒答"是明确事实)
    clean = pb.run_interleaved({"g1": good, "g2": good}, tasks=tasks, n=2)
    assert clean["refusals"] == {"g1": 0, "g2": 0}
    assert clean["refusal_rate"] == {"g1": 0.0, "g2": 0.0}


def test_divergence_threshold_boundary():
    """spread 恰好等于阈值时必须算分歧(>= 语义)。此前所有用例的 spread 都远离阈值,
    于是 >= 改成 > 在变异测试中存活 —— 边界相等是最容易漏测的一档。"""
    mk = lambda pid, p: [{"id": pid, "n": 4, "successes": int(p * 4), "pass_at_1": p,
                          "pass_hat_k": 0.0, "runs": [], "measured_at": 1.0}]
    reports = {"hi": mk("t", 1.0), "lo": mk("t", 0.75)}   # spread 恰为 0.25
    row = pb.reliability_matrix(reports, divergence=0.25, require_interleaved=False)[0]
    assert abs(row["spread"] - 0.25) < 1e-12
    assert row["divergent"] is True, "spread == divergence 必须判为分歧"
    # 略低于阈值则不算
    row2 = pb.reliability_matrix(reports, divergence=0.26, require_interleaved=False)[0]
    assert row2["divergent"] is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
