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



if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
