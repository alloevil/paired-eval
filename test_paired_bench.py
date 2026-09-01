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



if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
