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


def main(call=None):
    if call is None:
        print("!! 需注入模型调用: main(lambda prompt, : ...) 或在宿主环境提供 call")
        return 2
    problems = check_scaffold_effect(call)
    print()
    for p in problems:
        print(f"FAIL  {p}")
    if problems:
        print(f"\n{len(problems)} 项与记录不符 —— 先查是漂移还是环境变化, 再决定改记录还是改结论")
        return 1
    print("PASS  脚手架效应复现成功(方向、量级、显著性、有效样本四项均符合记录)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
