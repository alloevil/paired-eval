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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
