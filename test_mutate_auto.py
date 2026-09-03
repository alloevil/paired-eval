# -*- coding: utf-8 -*-
"""mutate_auto 的测试: 变异工具是 pre-push 门禁的承重件, 它的 bug 会
放过没测试的代码或拦住合法推送。三条作用域规则的 bug 此前全靠手工实验发现,
没有测试就会静默回归。运行: python3 test_mutate_auto.py

只测纯逻辑(降噪过滤/标识生成/diff解析/基线比对), 不跑真实变异(那需要几分钟)。
"""
import ast
import json
import pathlib
import tempfile

import mutate_auto as ma

SRC = '''
LIMIT = 100


def outer(a, timeout=30, cap=[5]):
    """默认参数与切片长度都是已知等价类。"""
    text = "abcdef"[:4]
    if a < 3 and a > 1:
        return text
    return None


class Holder:
    def method(self, x):
        return x <= 2 or x >= 9
'''


def test_noise_filter_skips_defaults_and_slices():
    """默认参数值与切片长度不该成为变异点: 它们是约定/消息长度, 不是契约。"""
    tree = ast.parse(SRC)
    all_pts = ma.count_points(ast.parse(SRC))
    filtered = ma.count_points(tree, ma._noise_constants(tree))
    assert filtered < all_pts, "降噪必须真的减少变异点"
    # 逐点收集标识, 确认 timeout=30 / cap=[5] / [:4] 都不在其中
    labels = []
    for i in range(filtered):
        t = ast.parse(SRC)
        m = ma._Mutator(i, ma._noise_constants(t), ma._scopes(t))
        m.visit(t)
        if m.applied:
            labels.append(m.applied)
    joined = " ".join(labels)
    for noise in ("30->31", "5->6", "4->5"):
        assert noise not in joined, f"已知等价类未被滤除: {noise} in {labels}"
    assert "100->101" in joined, "模块级普通常量仍应是变异点"


def test_labels_are_qualified_and_indexed():
    """标识 = 函数限定名 + 同类出现序号, 不含行号(行号会随无关插入平移)。"""
    tree = ast.parse(SRC)
    labels = []
    for i in range(ma.count_points(tree, ma._noise_constants(tree))):
        t = ast.parse(SRC)
        m = ma._Mutator(i, ma._noise_constants(t), ma._scopes(t))
        m.visit(t)
        labels.append(m.applied)
    assert any(l.startswith("outer#") for l in labels), f"应有 outer 作用域标识: {labels}"
    assert any(l.startswith("Holder.method#") for l in labels), f"应有嵌套限定名: {labels}"
    assert any(l.startswith("<module>#") for l in labels), f"模块级应标 <module>: {labels}"
    assert not any(l.startswith("L") and l[1:2].isdigit() for l in labels), \
        f"标识不得含行号: {labels}"
    # 同一函数内同类变异要有不同序号(outer 里有两个比较符)
    outer_cmp = [l for l in labels if l.startswith("outer#") and "->" in l]
    assert len(set(outer_cmp)) == len(outer_cmp), f"同类变异序号必须唯一: {outer_cmp}"


def test_label_stable_under_unrelated_insertion():
    """在上方插入无关代码后, 原有标识必须一字不变 —— 这正是弃用行号的原因。"""
    def labels_of(src):
        out = []
        for i in range(ma.count_points(ast.parse(src), ma._noise_constants(ast.parse(src)))):
            t = ast.parse(src)
            m = ma._Mutator(i, ma._noise_constants(t), ma._scopes(t))
            m.visit(t)
            if m.applied and not m.applied.startswith("<module>"):
                out.append(m.applied)
        return set(out)
    before = labels_of(SRC)
    after = labels_of("X = 7\nY = 8\n" + SRC)
    assert before <= after, f"原标识不得改变: 丢失 {before - after}"


def test_hunk_regex_parses_ranges():
    """diff hunk 头的解析: 单行(无逗号)与多行两种形态都要覆盖。"""
    diff = ("@@ -1 +1 @@\n" "@@ -10,0 +12,3 @@\n" "@@ -20,2 +30,1 @@\n")
    import re
    lines = set()
    for h in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff, re.M):
        start, count = int(h.group(1)), int(h.group(2) or 1)
        lines.update(range(start, start + count))
    assert lines == {1, 12, 13, 14, 30}, f"解析结果错误: {sorted(lines)}"


def _with_temp_baseline(fn):
    orig = ma.BASELINE
    with tempfile.TemporaryDirectory() as d:
        ma.BASELINE = pathlib.Path(d) / "baseline.json"
        try:
            return fn()
        finally:
            ma.BASELINE = orig


def test_baseline_scoped_to_modules_run():
    """只跑一个模块时, 其他模块的已知条目不得被当成"已修补"(否则更新基线会抹掉它们)。"""
    def body():
        ma.BASELINE.write_text(json.dumps({"survivors": ["a.py|f#0: X", "b.py|g#0: Y"]}),
                               encoding="utf-8")
        # 只跑 a.py 且其已知条目仍存活 -> 应通过, 且不提 b.py
        res = [("a.py", "f#0: X", False), ("a.py", "f#1: Z", True)]
        assert ma.check_against_baseline(res) == 0
        # 出现 a.py 的新存活 -> 失败
        res2 = [("a.py", "f#0: X", False), ("a.py", "f#9: NEW", False)]
        assert ma.check_against_baseline(res2) == 1
    _with_temp_baseline(body)


def test_partial_run_suppresses_fixed_and_refuses_write():
    """--since 只扫改动行: 同模块未扫到的条目会显示"已修补", 按提示更新就会抹掉它们。
    故 partial 下只报新增、且禁止写基线。"""
    def body():
        ma.BASELINE.write_text(json.dumps({"survivors": ["a.py|f#0: X", "a.py|h#0: W"]}),
                               encoding="utf-8")
        res = [("a.py", "g#0: Q", True)]          # 改动行里只有一个点, 且被杀死
        assert ma.check_against_baseline(res, partial=True) == 0, "partial 下应通过"
        assert ma.check_against_baseline(res, write=True, partial=True) == 1, \
            "partial 下写基线必须被拒绝"
        # 基线必须原封不动
        kept = json.loads(ma.BASELINE.read_text(encoding="utf-8"))["survivors"]
        assert kept == ["a.py|f#0: X", "a.py|h#0: W"], f"基线被改动: {kept}"
        # 全量写入则允许, 且只替换跑过模块的条目
        ma.BASELINE.write_text(json.dumps({"survivors": ["a.py|f#0: X", "b.py|g#0: Y"]}),
                              encoding="utf-8")
        assert ma.check_against_baseline([("a.py", "new#0: S", False)], write=True) == 0
        merged = json.loads(ma.BASELINE.read_text(encoding="utf-8"))["survivors"]
        assert merged == ["a.py|new#0: S", "b.py|g#0: Y"], f"合并错误: {merged}"
    _with_temp_baseline(body)


def test_missing_baseline_fails_check():
    """无基线时 --check 必须失败并给出指引, 而不是默认通过。"""
    def body():
        assert not ma.BASELINE.exists()
        assert ma.check_against_baseline([("a.py", "f#0: X", False)]) == 1
    _with_temp_baseline(body)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
