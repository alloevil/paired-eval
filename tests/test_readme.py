#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""README 里的 python 示例必须能对着真实 API 跑通 —— 文档里的代码也是代码, 且是最先腐坏的那种。

两个示例分别注入确定性假模型与假评委, 复刻真实失败机理(裸指令下加 markdown 围栏), 断言报告
与评分的关键字段。README.md 与 README.en.md 的示例块都跑(只有注释/提示词语言不同)。"""
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 项目根: 让 `python3 tests/x.py` 直接可跑
import re

import paired_bench as pb

ROOT = _pathlib.Path(__file__).resolve().parent.parent
_CANON = {t["instruction"]: t["canonical"] for t in pb.ALL_TASKS}


def _fake_llm(prompt):
    """带严格前缀(中英文示例各自的措辞)时答对; 裸指令时加围栏 -> 与实测的失败机理一致。"""
    body = next(k for k in _CANON if k in prompt)
    strict = prompt.startswith("严格") or prompt.startswith("Follow the format")
    return _CANON[body] if strict else "```json\n" + _CANON[body] + "\n```"


def _fake_judge(prompt, system, schema):
    if "claims" in str(schema):
        return {"claims": [{"text": "X 2023 年研发投入 640 亿元", "verifiable": True,
                            "importance": "core", "search_query": "X 2023 研发投入"}]}
    return {"verdict": "grounded", "reasoning": "r"}


def _blocks(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    return [b for b in re.findall(r"```python\n(.*?)```", text, re.S) if "my_llm" in b or "my_judge" in b]


def _run_examples(name):
    blocks = _blocks(name)
    assert len(blocks) == 2, f"{name}: 期望 2 个可执行示例(A/B 与 evaluate), 得 {len(blocks)}"
    ns = {"my_llm": _fake_llm, "my_judge": _fake_judge,
          "agent_answer": "X 2023 年研发投入 640 亿元。",
          "tool_observations": [{"tool_call_id": "t", "tool": "s", "observation": "X 2023 年研发投入 640 亿元。"}]}
    ab = blocks[0].replace("print(pb.report(", "_rp = (pb.report(")   # 抓住报告文本
    exec(compile(ab, f"{name}:ab", "exec"), ns)
    rp = ns["_rp"]
    assert "strict" in rp and "bare" in rp, rp
    # 假模型让 bare 在全部 4 题上因围栏失败 -> 强效应 + strict 触顶, 报告两者都要点出
    assert "触顶(1.000): strict" in rp, rp
    assert "显著" in rp, rp     # n=6 x 4 题 = 24 个不一致对, 远超 p 地板 -> 必须显著
    exec(compile(blocks[1], f"{name}:evaluate", "exec"), ns)
    r = ns["r"]
    assert r["score"] == 1.0 and r["metrics"]["fabricated"] == 0, r["metrics"]


def test_readme_zh_examples_run():
    _run_examples("README.md")


def test_readme_en_examples_run():
    _run_examples("README.en.md")


def test_readmes_cross_link_and_agree_on_counts():
    """两份 README 互相链接; 写死的数字(测试数、层数)与仓库实况一致 —— 否则第一句就是假的。"""
    zh = (ROOT / "README.md").read_text(encoding="utf-8")
    en = (ROOT / "README.en.md").read_text(encoding="utf-8")
    assert "README.en.md" in zh and "README.md" in en
    for name in ("findings", "lessons", "corrections"):
        assert f"docs/{name}.md" in zh and f"docs/{name}.md" in en, name
    # 统计原语表里列的函数都真的存在
    import claim_eval as ce
    for fn in ("paired_compare", "mcnemar_exact", "holm_adjust", "wilson_ci", "pass_hat_k", "required_tasks",
               "required_pairs", "detectable_effect", "p_floor", "min_units_for_alpha", "interpret"):
        assert callable(getattr(ce, fn)), fn
        assert f"`{fn}(" in zh and f"`{fn}(" in en, f"README 未列出 {fn}"
    for fn in ("saturation", "screen_tasks", "screen_graded", "report", "run_interleaved", "make_model"):
        assert callable(getattr(pb, fn)), fn
        assert fn in zh and fn in en, fn


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
