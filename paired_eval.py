# -*- coding: utf-8 -*-
"""paired_eval — 统一入口。仓库叫 paired-eval, 用户就该 `import paired_eval`。

本模块只做重导出, 不加语义: 各实现模块的名字(claim_eval / paired_bench / eval_task ...)是
历史形成的, 保留它们是为了不打断已有引用; 但新用户不该被迫先学这段历史才能找到入口。

    import paired_eval as pe
    pe.make_model / pe.run_interleaved / pe.report        # 配对 A/B 的三步
    pe.evaluate                                            # 按 task 的 verification.class 路由评分
    pe.paired_compare / pe.interpret / pe.required_pairs   # 统计原语
    pe.demo()                                              # 离线演示: 两个桩系统跑一遍完整报告

`python3 paired_eval.py` 直接跑 demo —— 不需要任何 API, 十秒内看到报告长什么样。
"""
from answer_match import extract_answer, grade_answer
from claim_eval import (Meter, detectable_effect, extract_claims, holm_adjust, interpret,
                        make_resilient, mcnemar_exact, min_units_for_alpha, p_floor,
                        paired_compare, pass_hat_k, required_pairs, required_tasks,
                        throttled_pmap, verify_trajectory, wilson_ci)
from eval_task import evaluate, evaluate_batch, evaluate_pair, repeat_evaluate, validate_task
from paired_bench import (ALL_TASKS, DERIVATION_CASES, RUBRIC_GATE, SCAFFOLD_SENSITIVE,
                          TRAJECTORY_GATE, make_model, pairwise_compare, reliability_matrix,
                          report, run_interleaved, run_paired, run_paired_repeated,
                          run_repeated, saturation, screen_graded, screen_tasks)
from rubric_eval import rubric_canary, run_rubric

__version__ = "0.1.0"
__all__ = [
    # 配对 A/B
    "make_model", "run_paired", "run_repeated", "run_paired_repeated", "run_interleaved",
    "reliability_matrix", "pairwise_compare", "report", "saturation", "screen_tasks", "screen_graded",
    # 任务与评分路由
    "evaluate", "evaluate_batch", "evaluate_pair", "repeat_evaluate", "validate_task",
    "ALL_TASKS", "SCAFFOLD_SENSITIVE", "DERIVATION_CASES", "TRAJECTORY_GATE", "RUBRIC_GATE",
    # 判定器
    "grade_answer", "extract_answer", "extract_claims", "verify_trajectory", "run_rubric", "rubric_canary",
    # 统计
    "paired_compare", "mcnemar_exact", "holm_adjust", "wilson_ci", "pass_hat_k",
    "required_tasks", "required_pairs", "detectable_effect", "p_floor", "min_units_for_alpha", "interpret",
    # 适配
    "Meter", "make_resilient", "throttled_pmap",
    "demo",
]


def demo(n=4, out=print):
    """离线演示: 两个桩系统在内置的 4 道格式敏感题上做配对 A/B, 打印完整报告。

    桩系统复刻一种真实观察到的失败形态 —— "bare" 在 JSON 题上把正确答案包进 markdown 围栏,
    内容对而格式坏(其余题两者相同, 用来展示"饱和题贡献零信息"的诊断); "strict" 直接给答案。
    没有模型调用, 只为展示报告的形状与读法。返回 report() 的结果 dict。"""
    canon = {t["instruction"]: t["canonical"] for t in ALL_TASKS}
    body = lambda p: canon[next(k for k in canon if k in p)]
    fenced = lambda p: ("```json\n" + body(p) + "\n```") if "JSON" in p else body(p)
    systems = {"strict": make_model(body), "bare": make_model(fenced)}
    tasks = [t for t in ALL_TASKS if t["id"] in SCAFFOLD_SENSITIVE]
    run = run_interleaved(systems, tasks=tasks, n=n, prompt_prefix="")
    rp = report(run["reports"], refusals=run["refusals"])
    out(rp["text"])
    return rp


if __name__ == "__main__":
    demo()
