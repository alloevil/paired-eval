#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""README 里的每个 python 代码块都必须能对着真实 API 跑通 —— 文档里的代码也是代码, 且是最先腐坏的那种。

两份 README 的代码块按出现顺序在同一个命名空间里依次执行。命名空间预先放好示例引用的外部对象
(answer / judge / scores_a / scores_b), 并把 examples.adapter_openai_compat 的 make_call / make_llm
换成确定性假实现 —— 假模型按任务 id 作答、按任务声明的答案标记组装, 因此中英两版共用同一套假对象。
README 里写死的事实(内置题数、链接目标、API 表里的名字)也在这里核对。"""
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 项目根: 让 `python3 tests/x.py` 直接可跑
import functools
import re

from paired_eval import claim_eval as ce
import paired_eval as pe
from paired_eval import rubric_eval as re_
from examples import adapter_openai_compat as ad

ROOT = _pathlib.Path(__file__).resolve().parent.parent


def _judge(prompt, system, schema):
    """假评委: 抽 claim / 判 grounding / 判 rubric 三种系统提示各给确定性答案。"""
    if system is ce.EXTRACT_SYSTEM:
        return {"claims": [{"text": "X 2023 revenue 41.2bn", "verifiable": True,
                            "importance": "core", "search_query": "X 2023 revenue"}]}
    if system is ce.JUDGE_TRAJ_SYSTEM:
        return {"verdict": "grounded", "source_tool_call": "t1", "evidence_quote": "q", "reasoning": "r"}
    assert system is re_.JUDGE_RUBRIC_SYSTEM, system
    return {"verdict": "met" if ("因为" in prompt or "because" in prompt) else "not_met", "reasoning": "r"}


def _fake_call_factory(ns):
    """make_call 的假实现。返回的 call 按提示里出现的任务指令定位任务, 按 verification 里的标记组装答案。
    model-a 解释过程且守标记; model-b 不解释(gated 题的 rubric 过不了); 裸指令(无严格前缀、非自检)下
    date 题不带标记 -> exact 判 not_attempted, 让 harness 与 agent 两个维度都有可观察的差异。"""
    def make_call(model=None, **kw):
        def call(prompt):
            tasks = ns.get("my_tasks") or []
            task = next((t for t in tasks if t["instruction"] in prompt), None)
            if task is None:                                     # 第一段示例的 task, 或找不到 -> 通用回答
                return "答案: 144, 因为 12 个 12 相加"
            v = task["verification"]
            marker = (v.get("gate") or v).get("marker", "答案")
            strict = prompt.startswith(ns.get("STRICT", "严格")) or "草稿" in prompt or "draft" in prompt
            if task["id"] == "date":
                return f"{marker}: 2024-03-05" if strict else "2024-03-05"
            if task["id"] == "sum-explained":
                why = ", 因为 12 个 12 相加 / because twelve twelves" if model != "model-b" else ""
                return f"{marker}: 144{why}"
            return task["observations"][0]["observation"]        # summary: 逐字复述资料 -> grounded
        return call
    return make_call


def _blocks(name):
    return re.findall(r"```python\n(.*?)```", (ROOT / name).read_text(encoding="utf-8"), re.S)


def _run(name):
    blocks = _blocks(name)
    assert len(blocks) >= 5, f"{name}: 期望至少 5 个 python 块, 得 {len(blocks)}"
    ns = {"judge": _judge, "scores_a": [1, 0, 1, 1, 0, 1, 1, 0], "scores_b": [1, 0, 0, 1, 0, 0, 1, 0]}
    ns["answer"] = "答案: 144, 因为 12 个 12 相加 / Answer: 144, because twelve twelves"
    printed = []
    ns["print"] = lambda *a, sep=" ", end="\n", **k: printed.append(sep.join(str(x) for x in a))
    real = (ad.make_call, ad.make_llm, pe.required_pairs)
    ad.make_call = _fake_call_factory(ns)
    ad.make_llm = lambda model=None, **kw: _judge
    pe.required_pairs = functools.partial(real[2], sims=6, resamples=300)   # 示例原样执行, 只调低蒙特卡洛精度
    try:
        for i, code in enumerate(blocks):
            exec(compile(code, f"{name}:block{i}", "exec"), ns)
    finally:
        ad.make_call, ad.make_llm, pe.required_pairs = real
    # 1) gated 示例: 答对且解释 -> 1.0, verdict None
    assert any(p.startswith("1.0 None") for p in printed), printed
    # 2) 三个对象各一份报告, 且每个维度都有可观察差异(不一致对不全为 0)
    reports = {p.split("\n", 1)[0]: p for p in printed if "\n" in p and ("不一致对" in p or "discordant" in p)}
    assert set(reports) == {"model", "harness", "agent"}, list(reports)
    for axis, text in reports.items():
        pairs = re.findall(r"(?:不一致对|discordant) (\d+):(\d+)", text)
        assert pairs and any(int(a) + int(b) > 0 for a, b in pairs), f"{axis} 维度应有不一致对: {text}"
    # model 维度: gated 题把"答对但不解释"的 model-b 判掉 -> A 胜
    assert re.search(r"(?:不一致对|discordant) (\d+):0", reports["model"]), reports["model"]
    # 3) 样本量块跑到了末行
    assert ns["pe"].interpret(ns["pe"].paired_compare(ns["scores_a"], ns["scores_b"]))["verdict"] == "null"
    assert len(ns["my_tasks"]) == 3 and {t["verification"]["class"] for t in ns["my_tasks"]} == {"exact", "gated", "trajectory"}
    return ns


def test_readme_zh_blocks_run():
    _run("README.zh-CN.md")


def test_readme_en_blocks_run():
    _run("README.md")


def test_demo_output_in_readme_matches_actual():
    """README 里贴的 demo 输出必须与 python3 -m paired_eval 的实际输出逐行一致 —— 贴假输出等于撒谎。"""
    for name, lang in (("README.zh-CN.md", None), ("README.md", "en")):
        lines = []
        pe.demo(out=lambda s: lines.extend(s.splitlines()), lang=lang)
        actual = "\n".join(lines)
        text = (ROOT / name).read_text(encoding="utf-8")
        assert actual in text, f"{name}: 贴的 demo 输出与实际不一致。实际:\n{actual}"
        assert "桩" in text or "stub" in text, f"{name}: 必须说明 demo 里是桩系统, 不是模型"


def test_readme_numbers_and_links_are_true():
    """README 写死的事实: 内置题数、互链、文档链接、API 表里的名字都真实存在。"""
    zh = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    en = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "README.md" in zh and "README.zh-CN.md" in en
    n = len(pe.ALL_TASKS)
    assert f"{n} 道中文冒烟题" in zh and f"{n} Chinese smoke tasks" in en, f"内置题数是 {n}"
    for path in ("docs/README.md", "docs/lessons.md", "docs/findings.md", "CONTRIBUTING.md",
                 "CHANGELOG.md", "examples/adapter_openai_compat.py"):
        assert path in zh and path in en, path
        assert (ROOT / path).exists(), path
    tables = (zh.split("## API 一览")[1].split("## 范围")[0], en.split("## API overview")[1].split("## Scope")[0])
    for table in tables:
        for name in set(re.findall(r"`([a-zA-Z_]+)`", table)):
            if name in ("import", "pe", "exact", "retrieval", "trajectory", "rubric", "gated"):
                continue
            assert name in pe.__all__ or hasattr(pe, name), f"API 表列了不存在的名字: {name}"
    assert not re.search(r"\d+ 个测试|\d+ tests", zh + en), "README 不写测试数 —— 它会过期(实测过一次)"
    # 三个评测对象与"先验后判"的原则必须出现在首屏(第一个 ## 之前或前两节)
    for text, words in ((zh, ("model", "harness", "agent", "gate")), (en, ("model", "harness", "agent", "gate"))):
        head = text.split("## ")[0] + "## ".join(text.split("## ")[1:3])
        for w in words:
            assert w in head, f"首屏缺 {w}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
