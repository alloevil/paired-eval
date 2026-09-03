#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发现复现检查: 把 paired_bench 头部记录的实测结论变成可执行断言。

为什么需要: 记录在 docstring 里的数字是历史声明, 无人能验证它今天是否还成立 ——
而本仓库第44轮的教训正是"结论会跨会话漂移"(同一模型同一题两个时间窗得 0/8 与 8/8)。
把最强的发现变成可复跑的检查, 漂移就会被发现而不是被继承。

刻意不叫 test_*.py: 需要真实模型调用(约32次), 放进 1.5 秒的快速套件会毁掉 pre-commit。
与 check_hooks.py 同层 —— 外层、慢、手动或 CI 跑。

用法: python3 reproduce_findings.py            用默认模型(需注入 model 调用)
      import reproduce_findings as rf; rf.main(call)   call(prompt, model) -> str
"""
import sys

import claim_eval as ce
import paired_bench as pb

STRICT = "严格按要求输出,不要任何多余内容。要求: "
BARE = ""

# 记录值来自 paired_bench 头部的 2x2 因子设计终报(第106轮)与其自我纠正(第107轮)。
# 容差留得宽: 复现的是"效应的存在与量级", 不是精确到小数点(那会因漂移而无谓失败)。
EXPECTED = {
    "scaffold_effect_min": 0.35,      # 记录值 +0.625; 低于 0.35 说明结论已漂移
    "scaffold_p_max": 0.05,           # 记录值 Holm 后 0.0117
    "informative_min": 2,             # 记录值 4/4 有信息; 少于 2 则样本已失效
    "model_null_bound": 0.10,         # 记录值: 模型效应 <10%(80 单元, MDE 0.10)
}


def check_scaffold_effect(call, n=4, verbose=True):
    """复现最强的那条: 固定模型, 严格脚手架 vs 裸指令, 在结构化输出题上。"""
    tasks = [t for t in pb.ALL_TASKS if t["id"] in pb.SCAFFOLD_SENSITIVE]
    assert len(tasks) >= 3, f"敏感子集缩水到 {len(tasks)} 题, 无法复现"
    mk = lambda pre: pb.make_model(lambda p, x=pre: call(x + p))
    out = pb.run_interleaved({"strict": mk(STRICT), "bare": mk(BARE)},
                             tasks=tasks, n=n, prompt_prefix="")
    rp = pb.report(out["reports"], refusals=out["refusals"])
    if verbose:
        print(rp["text"])
    pair = rp["pairs"][0]
    # 方向无关地取幅度: pairwise_compare 的 a/b 顺序按字典序
    effect = abs(pair["mean_diff"])
    sat = rp["saturation"]["informative"]
    p = pair["p_mcnemar_holm"]
    problems = []
    if effect < EXPECTED["scaffold_effect_min"]:
        problems.append(f"效应量 {effect:.3f} < 阈值 {EXPECTED['scaffold_effect_min']}"
                        f"(记录值 0.625) —— 结论可能已漂移")
    if p > EXPECTED["scaffold_p_max"]:
        problems.append(f"Holm 后 p={p:.5f} > {EXPECTED['scaffold_p_max']}(记录值 0.0117)")
    if sat < EXPECTED["informative_min"]:
        problems.append(f"有效样本 {sat} 题 < {EXPECTED['informative_min']} —— 题目已饱和, "
                        f"此检查失去意义(需换题而非改阈值)")
    # 方向必须是严格脚手架更好
    better = pair["a"] if pair["mean_diff"] > 0 else pair["b"]
    if better != "strict":
        problems.append(f"方向反转: {better} 反而更好 —— 这是重大发现, 不要当失败处理")
    return problems


def check_model_null(call_a, call_b, n=4, verbose=True):
    """复现 null 结论: 两模型在严格脚手架下不可区分(记录: 效应 <10%, 80 单元, MDE 0.10)。

    null 的复现比正向发现更容易骗人 —— 样本不足时"没测出差异"会被当成"复现成功"。
    故本函数把两件事分开返回:
      problems: null 已失效(差异变得可检出) —— 这是真正的漂移, 模型可能被更新了
      warnings: null 仍成立但这次的界比记录的松 —— 不是失败, 但不构成复现
    返回 (problems, warnings)。call_a/call_b 分别是两个被比模型的调用。
    """
    tasks = [t for t in pb.ALL_TASKS if t["id"] in pb.SCAFFOLD_SENSITIVE]
    mk = lambda f: pb.make_model(lambda p, g=f: g(STRICT + p))
    out = pb.run_interleaved({"a": mk(call_a), "b": mk(call_b)},
                             tasks=tasks, n=n, prompt_prefix="")
    rp = pb.report(out["reports"], refusals=out["refusals"])
    if verbose:
        print(rp["text"])
    pair = rp["pairs"][0]
    verdict = pair["interpretation"]
    problems, warnings = [], []
    if verdict["verdict"] == "significant":
        problems.append(f"null 已失效: 模型差异变得可检出(Δ={pair['mean_diff']:+.3f}, "
                        f"p={pair['p_mcnemar_holm']:.5f}) —— 记录值是『效应 <10%』。"
                        f"查是模型被更新还是题目改变, 不要直接改记录")
        return problems, warnings
    bound = verdict["rules_out"]
    if bound is None:
        warnings.append(f"这次的 null 无信息: {verdict['n_units']} 个单元下 MDE 不可达 —— "
                        f"不能算复现(记录值是 80 单元 / MDE 0.10)")
    elif bound > EXPECTED["model_null_bound"]:
        warnings.append(f"null 仍成立但界更松: 只排除 >={bound:.0%}(记录 "
                        f"<={EXPECTED['model_null_bound']:.0%}) —— 需 "
                        f"{ce.required_tasks(EXPECTED['model_null_bound'], 0.0, sims=200)} "
                        f"个单元才能复现原界")
    return problems, warnings


def main(call=None, call_b=None, n=4):
    """跑全部复现检查。call: 主模型; call_b: 第二模型(给出才跑 null 复现)。
    退出码 0=全部复现(warnings 不算失败), 1=有结论与记录不符, 2=未注入调用。"""
    if call is None:
        print("!! 需注入模型调用: main(lambda prompt: ...) 或在宿主环境提供 call")
        return 2
    print("=== 1 脚手架效应(正向发现) ===")
    problems = check_scaffold_effect(call, n=n)
    warnings = []
    passed = ["脚手架效应"] if not problems else []
    if call_b is not None:
        print("\n=== 2 模型 null(记录: 效应 <10%) ===")
        p2, w2 = check_model_null(call, call_b, n=n)
        problems += p2
        warnings += w2
        if not p2:
            passed.append("模型 null" + (" (界更松)" if w2 else ""))
    print()
    for w in warnings:
        print(f"WARN  {w}")
    for p in problems:
        print(f"FAIL  {p}")
    if problems:
        print(f"\n{len(problems)} 项与记录不符 —— 先查是漂移还是环境变化, 再决定改记录还是改结论")
        return 1
    print(f"PASS  {', '.join(passed)} 复现成功"
          + ("(见上方 WARN: 界比记录松, 不构成完整复现)" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
