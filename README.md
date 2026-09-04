# paired-eval

比较两个 LLM 系统——不同的模型、提示、脚手架或 agent 策略——**哪个更好，这个结论有多可靠，样本够不够**。

[English](README.en.md) · [文档索引](docs/README.md) · [更新日志](CHANGELOG.md)

[![CI](https://github.com/alloevil/paired-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/alloevil/paired-eval/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

纯标准库 Python ≥ 3.9，零第三方依赖。不绑定任何模型供应商——模型调用由你以 `call(prompt) -> str` 的形式注入。

## 为什么需要它

评测两个 LLM 系统时最常见的三个错误，本项目各用一个机制挡住：

| 常见做法 | 问题 | 本项目的做法 |
|---|---|---|
每个系统各跑一遍，比平均分 | 单次运行方差大；两个独立均值的差既不配对也没有区间 | **同题交错重复**，逐题配对，报效应量与 bootstrap 区间 |
用 t 检验比两列分数 | 分数不是正态的，样本通常只有几十题 | **McNemar 精确检验**（二值）与**符号翻转置换检验**（连续分），多重比较做 Holm 校正 |
"p > 0.05，两者没有差别" | "不显著"有三种完全不同的含义 | **`interpret()` 把结果翻译成四种结论之一**：显著 / 有界的 null（能排除多大效应）/ 无信息 / 检验无力（样本再多效应也测不出，并告诉你缺什么） |

## 安装

```sh
pip install git+https://github.com/alloevil/paired-eval.git
```

或者 clone 后直接 `import`——没有依赖，不需要构建。

## 十秒钟看效果（不需要 API）

```sh
python3 paired_eval.py
```

两个桩系统在 4 道内置格式题上做配对 A/B（`bare` 在 JSON 题上把正确答案包进 markdown 围栏），实际输出：

```
有效样本: 2/4 题有信息(恒过 2, 恒败 0)
拒答: {'strict': 0, 'bare': 0}
触顶(1.000): strict —— 该系统无余量, 它作为参照时效应会被压缩
bare vs strict: Δ=-0.500 CI95=[-1.000,+0.000] | 逐题p=0.506 逐轮McNemar=0.00781 Holm=0.00781 | 不一致对 0:8 集中度=0.50 | 显著: Δ=-0.500 CI95=[-1.000,+0.000] p=0.0078 (n=16)
```

怎么读：4 题里 2 题两系统都全对，**不携带任何区分信息**（有效样本 2/4）；`strict` 满分，作为参照会压缩效应；
8 个不一致对全部偏向 `strict`，且分布在 2 题上（集中度 0.50，不是单题异常）；逐题置换检验 p=0.506 是因为只有 2 道有信息题——
**它的最小可能 p 就是 0.5**，而逐轮 McNemar 用上了 16 个配对单元，p=0.0078。最后一段是 `interpret()` 给出的可直接写进报告的结论。

## 用你自己的系统和任务

**1. 定义任务。** 一个任务是一个 dict：指令、程序判定器、一个合法输出样例（自检用）。判定器故意严格——测的就是指令遵循。

```python
import paired_eval as pe

tasks = [
    {"id": "date-iso", "instruction": "把 2024年3月5日 写成 ISO 8601 日期, 只输出日期",
     "check": lambda r: r.strip() == "2024-03-05", "canonical": "2024-03-05"},
    {"id": "json-pair", "instruction": '输出 JSON 对象 {"a": 1}, 不要其他内容',
     "check": lambda r: r.strip() == '{"a": 1}', "canonical": '{"a": 1}'},
]
```

**2. 接上模型。** 任何 `call(prompt) -> str` 都行；`make_model` 负责有界重试，失败转为"拒答"成对丢弃而不崩整批。
[`examples/adapter_openai_compat.py`](examples/adapter_openai_compat.py) 是一个只用标准库的 OpenAI 兼容适配器。

```python
systems = {"strict": pe.make_model(call_strict), "bare": pe.make_model(call_bare)}
```

**3. 跑，然后读报告。** `n` 是每题每系统的重复次数；交错运行会轮转先后顺序，避免时间窗漂移进入比较。

```python
run = pe.run_interleaved(systems, tasks=tasks, n=8, prompt_prefix="")
print(pe.report(run["reports"], refusals=run["refusals"])["text"])
```

报告里的每一项都有它防的错：

| 字段 | 含义 |
|---|---|
有效样本 | 两系统全轮次同结果的题不贡献信息；**MDE 要按有信息题数读** |
拒答 | 各系统被丢弃的调用数——拒答率本身是结果的一部分 |
触顶 / 触底 | 满分或零分的系统无余量，与它比较时效应被压缩，交互项不可解释 |
Δ, CI95 | 效应量与 bootstrap 95% 区间 |
逐题 p / 逐轮 McNemar / Holm | 两种配对检验；多系统两两比较时的多重校正 |
不一致对 a:b, 集中度 | 分歧的方向与分布；集中度 1.0 = 全部来自一道题 |
结论句 | `interpret()` 的四种判定之一，null 时附"能排除多大效应"或"缺多少单元" |

## 还能做什么

**评 agent 的回答是否忠实于它看到的观察**（逐条抽 claim，对轨迹 grounding，区分 grounded / distorted / fabricated）：

```python
r = pe.evaluate({"id": "q1", "instruction": "总结资料",
                 "verification": {"class": "trajectory", "grounding_policy": "must_ground"}},
                response=answer, observations=observations, llm=judge)
print(r["score"], r["metrics"]["fabricated"])      # grounding_rate, 无据 claim 数
```

`judge(prompt, system, schema) -> dict` 是带 JSON-schema 约束的结构化调用；`evaluate` 还支持 `exact`（纯程序，无 LLM）、
`retrieval`（检索核实）与 `rubric`（逐条二值判定 + 加权，`rubric_canary` 检验 rubric 本身是否会被糊弄回答骗过）。

**在跑之前算样本量**，而不是跑完碰运气：

```python
pe.required_tasks(0.30, 0.10)         # A 胜率 30%、负率 10% 时, 要多少配对任务才有 80% 功效
pe.required_pairs(0.15, 0.30)         # 连续分数版: 均值差 0.15、sd 0.30 -> 内部真跑置换检验求 n
pe.detectable_effect(31)              # 反过来: 31 道题最小能检出多大的单方面胜率
pe.interpret(pe.paired_compare(scores_a, scores_b))["text"]   # 事后: 这个结果能说什么
```

**筛出有区分力的题**：`screen_tasks` / `screen_graded` 两阶段筛选（筛 + 复核），默认排除成功率 > 0.9 的近顶题——它们在任何可负担轮数下都筛不稳。

## API 一览

`import paired_eval as pe` 即可用到全部入口（实现分布在几个模块里，文件名是历史形成的）：

| 分组 | 入口 |
|---|---|
配对 A/B | `make_model` `run_interleaved` `run_paired` `run_repeated` `report` `pairwise_compare` `reliability_matrix` `saturation` |
任务与评分 | `evaluate` `evaluate_pair` `evaluate_batch` `validate_task` · 内置 `ALL_TASKS`（31 道中文冒烟题，示例与自测用） |
判定器 | `grade_answer`（数值/选项/集合/`\boxed{}`）`extract_claims` `verify_trajectory` `run_rubric` `rubric_canary` |
统计 | `paired_compare` `mcnemar_exact` `holm_adjust` `wilson_ci` `pass_hat_k` `required_tasks` `required_pairs` `detectable_effect` `p_floor` `min_units_for_alpha` `interpret` |
筛题 | `screen_tasks` `screen_graded` |
适配 | `make_resilient`（有界重试）`throttled_pmap`（限并发）`Meter`（成本计量） |

每个函数的 docstring 都说明了它防的是什么错；方法学背景见 [docs/lessons.md](docs/lessons.md)。

## 范围与状态

- **是**：比较两个（或多个）系统的统计工具，加上一套可组合的验证器（精确匹配 → 程序判定 → 检索核实 → judge / rubric）。
- **不是**：大规模题库（内置 31 题只作示例与自测，且为中文）、评测平台或仪表盘、模型供应商客户端。
  跑大规模任务集请用专门的评测框架；把它们输出的**逐题分数**喂给 `paired_compare` / `interpret`，本项目回答的是"这个差别可信吗、还差多少样本"。
- **状态**：0.1.0，单作者，接口可能变。代码、文案与内置题为中文；报告文本目前只有中文。
- [docs/findings.md](docs/findings.md) 是用本工具做的一次案例研究（一对特定模型、三族任务）；数字是实例特定的，展示的是报告该怎么写，不是可引用的普适结论。

## 文档 · 贡献 · 许可

[docs/README.md](docs/README.md) 是文档索引：方法学教训、案例研究、被推翻的结论。
贡献前请读 [CONTRIBUTING.md](CONTRIBUTING.md)（钩子、变异测试基线、统计代码的额外纪律）。MIT 许可。
