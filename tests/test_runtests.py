# -*- coding: utf-8 -*-
"""runtests.sh 的测试: 它是所有门禁的基座(pre-commit/pre-push 都调它),
自身的结构自检此前只做过手工验证 —— 若被"简化"掉, 静默跳过的测试会重新出现。
运行: python3 tests/test_runtests.py

做法: 在临时目录里复制 runtests.sh, 造各种形态的合成测试文件, 校验退出码与输出。
runtests.sh 会 cd 到自己所在目录, 因此拷贝即隔离, 不会碰真实套件。
"""
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 项目根: 让 `python3 tests/x.py` 直接可跑
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent   # 仓库根(tests/ 的上一级)

# 合成测试文件用的 main 块。这里刻意拼接字符串: 若字面写出 `if __name__`,
# runtests.sh 的结构自检会把本文件误计为"两个 main 块"(它按行首正则数)。
# 这正是被自己的门禁抓住一次的结果 —— 门禁宁可对自己也严格。
_IF = "if " + "__name__" + ' == "__main__":'
MAIN = f'''

{_IF}
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {{t.__name__}}")
    print(f"\\n{{len(tests)}} tests passed")
'''


def _run(files):
    """在隔离目录里跑 runtests.sh, 返回 (退出码, 输出)。"""
    with tempfile.TemporaryDirectory() as d:
        dst = pathlib.Path(d)
        shutil.copy(ROOT / "runtests.sh", dst / "runtests.sh")
        (dst / "tests").mkdir()
        for name, body in files.items():
            (dst / "tests" / name).write_text(body, encoding="utf-8")
        proc = subprocess.run(["sh", str(dst / "runtests.sh")],
                              capture_output=True, text=True)
        cache_dirs = list(dst.glob("__pycache__"))
        return proc.returncode, proc.stdout + proc.stderr, cache_dirs


def test_wellformed_suite_passes():
    code, out, _ = _run({"test_ok.py": "def test_a():\n    assert True\n" + MAIN})
    assert code == 0, f"合规套件应通过: {out}"
    assert "1 tests passed" in out


def test_failing_assertion_fails():
    code, out, _ = _run({"test_bad.py": "def test_a():\n    assert False, '故意失败'\n" + MAIN})
    assert code != 0 and "FAILED" in out, f"断言失败必须非零退出: {out}"


def test_test_after_main_is_caught():
    """定义在 __main__ 块之后的测试永远不会被执行, 套件却会显示全绿 ——
    这是本仓库踩过两次的陷阱, 结构自检必须抓住(定义数 != 执行数)。"""
    body = "def test_a():\n    assert True\n" + MAIN + "\n\ndef test_skipped():\n    assert False\n"
    code, out, _ = _run({"test_skip.py": body})
    assert code != 0, f"静默跳过必须被抓住: {out}"
    assert "静默跳过" in out and "定义了 2 个测试" in out, out


def test_duplicate_main_block_is_caught():
    body = "def test_a():\n    assert True\n" + MAIN + MAIN
    code, out, _ = _run({"test_dup.py": body})
    assert code != 0 and "__main__ 块数量为 2" in out, out


def test_shadowed_test_is_caught():
    """同名函数会被后定义者覆盖, 执行数因此少于定义数。"""
    body = ("def test_a():\n    assert True\n\n\n"
            "def test_a():\n    assert True\n" + MAIN)
    code, out, _ = _run({"test_shadow.py": body})
    assert code != 0 and "静默跳过" in out, out


def test_no_bytecode_cache_written():
    """必须用 python3 -B: pyc 只按 (源mtime秒级, size) 验缓存, 同大小的改动
    若发生在同一秒内会命中过期字节码, 门禁给出假绿(本仓库做变异测试时实测踩到过)。
    注意: 直接执行的脚本本身不写 pyc, 只有被 import 的模块才会 ——
    所以合成用例必须 import 一个本地模块, 否则这条测试是伪保护(初版就是)。"""
    code, out, caches = _run({
        "helper.py": "VALUE = 1\n",
        "test_ok.py": "import helper\n\n\ndef test_a():\n    assert helper.VALUE == 1\n" + MAIN,
    })
    assert code == 0, out
    assert not caches, f"不应生成 __pycache__(需 python3 -B): {caches}"


def test_multiple_files_all_reported():
    code, out, _ = _run({
        "test_one.py": "def test_a():\n    assert True\n" + MAIN,
        "test_two.py": "def test_b():\n    assert True\n\n\ndef test_c():\n    assert True\n" + MAIN,
    })
    assert code == 0 and "[test_one.py] 1 tests passed" in out and \
        "[test_two.py] 2 tests passed" in out, out




def test_summary_line_is_last_and_tail_safe():
    """末行必须是汇总, 且成败都在其中 —— 逐文件输出被 tail 截断时靠它兜底。
    真实踩坑(docs/lessons.md#tail-safe): 两个套件已红, 我看 `| tail -3` 全绿就继续往下做了。"""
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        shutil.copy(ROOT / "runtests.sh", work / "runtests.sh")
        # 全绿: 末行给出测试总数
        (work / "tests").mkdir()
        (work / "tests" / "test_a.py").write_text(
            'def test_one():\n    pass\n\n\nif __name__ == "__main__":\n'
            '    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]\n'
            "    for t in tests:\n        t()\n"
            '    print(f"\\n{len(tests)} tests passed")\n', encoding="utf-8")
        r = subprocess.run(["sh", "runtests.sh"], cwd=work, capture_output=True, text=True)
        last = r.stdout.strip().splitlines()[-1]
        assert r.returncode == 0 and last == "=== 全部通过: 1 测试 ===", last
        # 加一个失败文件, 并让它在字母序上排在前面 -> 逐文件的 FAILED 会被 tail 截掉
        (work / "tests" / "test_0bad.py").write_text(
            'def test_bad():\n    assert False\n\n\nif __name__ == "__main__":\n'
            '    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]\n'
            "    for t in tests:\n        t()\n"
            '    print(f"\\n{len(tests)} tests passed")\n', encoding="utf-8")
        r = subprocess.run(["sh", "runtests.sh"], cwd=work, capture_output=True, text=True)
        lines = r.stdout.strip().splitlines()
        assert r.returncode != 0
        assert lines[-1] == "=== 套件失败: test_0bad.py ===", lines[-1]
        # 关键性质: 只看最后两行也能发现失败(这正是当初漏掉的场景)
        assert any("失败" in l for l in lines[-2:]), lines[-2:]
        # 失败文件名必须点出来, 否则还得重跑一次才知道是谁
        assert "test_0bad.py" in lines[-1] and "test_a.py" not in lines[-1]




def test_empty_suite_is_a_failure():
    """tests/ 下一个测试文件都没有时必须失败 —— 空套件看起来像"全部通过: 0 测试",
    而 pre-commit 只看退出码。目录搬迁(平铺 -> tests/)时 glob 写错就会撞上这种静默通过。"""
    code, out, _ = _run({})
    assert code != 0 and "未找到任何 test_*.py" in out, out


def test_docs_anchors_resolve():
    """代码注释与文档里的 docs/<name>.md#<anchor> 引用必须都能解析到 <a name> 定义。
    实验日志迁出源码后, 代码里只留锚点 —— 锚点悬空等于把"为什么"这一层弄丢了。"""
    import re
    defined = {}
    for md in (ROOT / "docs").glob("*.md"):
        defined[md.name] = set(re.findall(r'<a name="([^"]+)"></a>', md.read_text(encoding="utf-8")))
    assert defined, "docs/ 下应有带锚点的 markdown"
    refs, missing = 0, []
    scan = list(ROOT.glob("*.py")) + list(ROOT.glob("*.sh")) + list((ROOT / "tests").glob("*.py")) \
        + list((ROOT / "docs").glob("*.md")) + list(ROOT.glob("*.md"))
    for f in scan:
        for doc, anchor in re.findall(r'(findings|lessons|corrections)\.md#([a-z0-9-]+)',
                                      f.read_text(encoding="utf-8")):
            refs += 1
            if anchor not in defined.get(f"{doc}.md", set()):
                missing.append(f"{f.relative_to(ROOT)}: {doc}.md#{anchor}")
    assert refs >= 40, f"锚点引用数异常少({refs}), 迁移可能没生效"
    assert not missing, "悬空锚点:\n  " + "\n  ".join(missing)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
