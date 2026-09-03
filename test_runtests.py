# -*- coding: utf-8 -*-
"""runtests.sh 的测试: 它是所有门禁的基座(pre-commit/pre-push 都调它),
自身的结构自检此前只做过手工验证 —— 若被"简化"掉, 静默跳过的测试会重新出现。
运行: python3 test_runtests.py

做法: 在临时目录里复制 runtests.sh, 造各种形态的合成测试文件, 校验退出码与输出。
runtests.sh 会 cd 到自己所在目录, 因此拷贝即隔离, 不会碰真实套件。
"""
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent

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
        for name, body in files.items():
            (dst / name).write_text(body, encoding="utf-8")
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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
