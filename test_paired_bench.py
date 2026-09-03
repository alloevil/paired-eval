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



if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
