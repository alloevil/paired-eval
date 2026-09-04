#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""README 里的每个 python 代码块都必须能对着真实 API 跑通 —— 文档里的代码也是代码, 且是最先腐坏的那种。

两份 README 的代码块按出现顺序在同一个命名空间里依次执行, 命名空间预先放好示例引用的外部对象
(call_strict / call_bare / answer / observations / judge / scores_a / scores_b), 全部是确定性假对象。
README 里写死的数字(内置题数)也在这里核对, 否则第一句就可能是假的。"""
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 项目根: 让 `python3 tests/x.py` 直接可跑
import functools
import re

import paired_eval as pe

ROOT = _pathlib.Path(__file__).resolve().parent.parent


def _fakes():
    """示例引用的外部对象。call_bare 复刻真实失败形态: JSON 题上把正确答案包进围栏。"""
    answers = {"date-iso": "2024-03-05", "json-pair": '{"a": 1}'}
    instr = {}   # instruction -> answer, 由 tasks 块执行后在 ns 里填(见 _run)

    def pick(prompt):
        return next(v for k, v in instr.items() if k in prompt)

    def call_strict(prompt):
        return pick(prompt)

    def call_bare(prompt):
        a = pick(prompt)
        return ("```json\n" + a + "\n```") if "JSON" in prompt else a

    def judge(prompt, system, schema):
        if "claims" in str(schema):
            return {"claims": [{"text": "X spent 64bn on R&D in 2023", "verifiable": True,
                                "importance": "core", "search_query": "X 2023 R&D"}]}
        return {"verdict": "grounded", "reasoning": "r"}

    return {"_answers": answers, "_instr": instr, "call_strict": call_strict, "call_bare": call_bare,
            "judge": judge, "answer": "X spent 64bn on R&D in 2023.",
            "observations": [{"tool_call_id": "t", "tool": "s", "observation": "X spent 64bn on R&D in 2023."}],
            "scores_a": [1, 0, 1, 1, 0, 1, 1, 0], "scores_b": [1, 0, 0, 1, 0, 0, 1, 0]}


def _blocks(name):
    return re.findall(r"```python\n(.*?)```", (ROOT / name).read_text(encoding="utf-8"), re.S)


def _run(name):
    blocks = _blocks(name)
    assert len(blocks) >= 5, f"{name}: 期望至少 5 个 python 块, 得 {len(blocks)}"
    ns = _fakes()
    printed = []
    ns["print"] = lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    # required_pairs 内部真跑蒙特卡洛(~20s): 示例代码原样执行, 但把默认精度调低 —— 测的是 API 契约, 不是那个数
    real_rp = pe.required_pairs
    pe.required_pairs = functools.partial(real_rp, sims=6, resamples=300)
    try:
        for i, code in enumerate(blocks):
            exec(compile(code, f"{name}:block{i}", "exec"), ns)
            if "tasks" in ns and not ns["_instr"]:        # tasks 块刚执行: 把指令映射到答案, 供假模型用
                ns["_instr"].update({t["instruction"]: ns["_answers"][t["id"]] for t in ns["tasks"]})
    finally:
        pe.required_pairs = real_rp
    # A/B 块: bare 只在 JSON 题上失败 -> 1/2 有信息, strict 触顶, 8 个不一致对全偏 strict
    rp = [p for p in printed if "不一致对" in p]
    assert len(rp) == 1, printed
    text = rp[0]
    assert "有效样本: 1/2" in text and "触顶(1.000): strict" in text, text
    assert "不一致对 0:8" in text and "显著" in text, text
    # evaluate 块
    assert any(p.startswith("1.0 0") for p in printed), printed
    # 样本量块: 四行都执行到了(命名空间里留下的表达式不留痕, 用副作用检查最后一行)
    assert ns["pe"].interpret(ns["pe"].paired_compare(ns["scores_a"], ns["scores_b"]))["verdict"] == "null"
    return ns


def test_readme_zh_blocks_run():
    _run("README.md")


def test_readme_en_blocks_run():
    _run("README.en.md")


def test_demo_output_in_readme_matches_actual():
    """README 里贴的 demo 输出必须与 python3 paired_eval.py 的实际输出逐行一致 —— 贴假输出等于撒谎。"""
    for name, lang in (("README.md", None), ("README.en.md", "en")):
        lines = []
        pe.demo(out=lambda s: lines.extend(s.splitlines()), lang=lang)
        actual = "\n".join(lines)
        text = (ROOT / name).read_text(encoding="utf-8")
        assert actual in text, f"{name}: 贴的 demo 输出与实际不一致。实际:\n{actual}"


def test_readme_numbers_and_links_are_true():
    """README 写死的事实: 内置题数、互链、文档链接、API 表里的名字都真实存在。"""
    zh = (ROOT / "README.md").read_text(encoding="utf-8")
    en = (ROOT / "README.en.md").read_text(encoding="utf-8")
    assert "README.en.md" in zh and "README.md" in en
    n = len(pe.ALL_TASKS)
    assert f"{n} 道中文冒烟题" in zh and f"{n} Chinese smoke tasks" in en, f"内置题数是 {n}"
    for path in ("docs/README.md", "docs/lessons.md", "docs/findings.md", "CONTRIBUTING.md",
                 "CHANGELOG.md", "examples/adapter_openai_compat.py"):
        assert path in zh and path in en, path
        assert (ROOT / path).exists(), path
    # API 表里出现的每个反引号名字都必须是 paired_eval 的真实导出
    tables = (zh.split("## API 一览")[1].split("## 范围")[0], en.split("## API overview")[1].split("## Scope")[0])
    for table in tables:
        for name in set(re.findall(r"`([a-zA-Z_]+)`", table)):
            if name in ("import", "pe"):
                continue
            assert name in pe.__all__ or hasattr(pe, name), f"API 表列了不存在的名字: {name}"
    assert not re.search(r"\d+ 个测试|\d+ tests", zh + en), "README 不写测试数 —— 它会过期(实测过一次)"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
