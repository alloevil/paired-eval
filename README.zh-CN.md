<h1 align="center">paired-eval</h1>

**paired-eval** 是一个零依赖的 Python 库，替已经拿到逐题 A/B 结果的工程师回答：一个模型、一个 agent 策略、一套 harness，是否真的比另一个更好。

<p align="center">
  <a href="https://alloevil.github.io/paired-eval/"><img src="docs/assets/logo.svg" width="96" height="96" alt="paired-eval"></a>
</p>
<p align="center"><em>给模型、agent、harness 做评测：能用程序验证的先验，验不过的再交给 rubric；每一次比较都做成统计上诚实的配对。</em></p>
<p align="center">
  <a href="https://github.com/alloevil/paired-eval/actions/workflows/ci.yml"><img src="https://github.com/alloevil/paired-eval/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/dependencies-none-brightgreen.svg" alt="Dependencies: none"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License: MIT"></a>
</p>
<p align="center"><a href="https://alloevil.github.io/paired-eval/">主页</a> · <a href="README.md">English</a> · <a href="docs/README.md">文档</a> · <a href="CHANGELOG.md">更新日志</a></p>

纯标准库 Python ≥ 3.9，零依赖。模型与评委都由你注入（`call(prompt) -> str`、`judge(prompt, system, schema) -> dict`），不绑定供应商。

> **同名，但不是同一个项目。** 本项目就是发布在 [PyPI](https://pypi.org/project/paired-eval/) 上的 `paired-eval`，源码在本仓库——`pip install paired-eval`，导入名 `import paired_eval`。另有一个同名的无关项目 [labsyspharm/paired-eval](https://github.com/labsyspharm/paired-eval)，按它自己的 README 是"paired evaluation 的 O(n log n) 实现"，用于机器学习预测值与真实标签的配对比较，从 GitHub 安装、导入名为 `pairedeval`。两者互不为分支，也不是同一项目的不同版本。

## 它是什么

一套把两个系统比出可信结论的方法与小工具：验证可以叠加——有唯一真值的交给程序（program gate），必须忠于资料的逐条 claim 对观察做 grounding，只剩质量标准的才交给 rubric——然后把逐题分数变成配对统计的结论。它**不是**大规模题库、评测平台、模型客户端，也不是 rubric 生成器：模型与评委由你注入，内置 31 题只作示例与自测。

## 它评什么

三个对象问的是三个不同的问题，各自固定不同的变量；混着评什么都得不到。

| 评什么 | 固定什么 | 换什么 | 一次 A/B 长什么样 |
|---|---|---|---|
**model** | 同一批题、同一脚手架 | 模型 | `{"A": call_a, "B": call_b}` |
**harness** | 同一模型 | 提示 / 脚手架 / 工具接线 | `{"strict": 加严格前缀, "bare": 裸指令}` |
**agent** | 同一模型、同一脚手架 | 策略 | `{"single": 单遍, "self-check": 出草稿后自检修正}` |
**三者的交互** | 2×2 因子设计 | 两个因子同时换 | 同一把尺子上比主效应，并自动报出天花板 |

## 怎么验：能用程序验的先验，验不过的再判

```
有唯一真值 ──→ exact(数值 / 选项 / 集合 / \boxed{}) 或程序判定 check(response) -> bool
只有事实来源 ──→ retrieval(检索核实) / trajectory(逐条 claim 对 agent 看到的观察做 grounding)
只有质量标准 ──→ rubric(逐条二值判定 + 加权; rubric 自身要过 canary: 糊弄回答不得高分)
```

层可以叠加——**程序做 gate，rubric 做 score**。验不过直接 0 分且不调用评委：不为一个已经错了的回答付 judge 的钱，也不让 judge 在"能不能跑"上被表面功夫骗过，它只在确实过关的候选里分高下。

```python
import paired_eval as pe

task = {"id": "sum-explained", "instruction": "12×12 等于多少, 并说明过程; 以「答案: 」开头给出结果",
        "verification": {"class": "gated",
                         "gate":  {"class": "exact", "gold": "144", "kind": "numeric"},
                         "score": {"class": "rubric", "criteria": [{"text": "说明了计算过程", "weight": 1}]}}}
r = pe.evaluate(task, response=answer, llm=judge)
print(r["score"], r["verdict"])        # 答错 -> 0.0 'gated_out'(judge 一次都没调用); 答对 -> rubric 分
```

`evaluate` 的其他路由：`exact`（纯程序，无 LLM）、`retrieval`、`trajectory`、`rubric`；`rubric_canary` 检验一份 rubric 会不会被糊弄回答骗过。

## 安装

```sh
pip install paired-eval
```

Python ≥ 3.9，零第三方包——`pyproject.toml` 里 `dependencies = []` 是设计约束。模型与评委由你注入（`call(prompt) -> str`、`judge(prompt, system, schema) -> dict`），只用标准库的 OpenAI 兼容适配器见 [examples/adapter_openai_compat.py](examples/adapter_openai_compat.py)。想不接 API 先看统计层，跑 `python3 -m paired_eval`。

## 十秒钟看效果（统计层的离线演示）

```sh
python3 -m paired_eval            # --lang en 输出英文报告
```

**这里的 `strict` / `bare` 是两个桩函数，不是模型**——它们复刻一种真实观察到的失败机理（裸指令下正确答案被包进 markdown 围栏，JSON 解析失败），只为在没有 API 的情况下展示报告的形状。输出是真的（由测试与当前代码逐行比对）：

```
有效样本: 2/4 题有信息(恒过 2, 恒败 0)
拒答: {'strict': 0, 'bare': 0}
触顶(1.000): strict —— 该系统无余量, 它作为参照时效应会被压缩
bare vs strict: Δ=-0.500 CI95=[-1.000,+0.000] | 逐题p=0.506 逐轮McNemar=0.00781 Holm=0.00781 | 不一致对 0:8 集中度=0.50 | 显著: Δ=-0.500 CI95=[-1.000,+0.000] p=0.0078 (n=16)
```

两题两系统全对、不携带信息（有效样本 2/4）；8 个不一致对全部偏向 `strict`，分布在 2 题上（集中度 0.50）；逐题置换 p=0.506 是因为只有 2 道有信息题，**它的最小可能 p 就是 0.5**，而逐轮 McNemar 用上全部 16 个配对单元。末段是可直接写进报告的结论。

## 用真实模型：三个对象各做一次 A/B

**1. 任务集按"有什么可验"混搭验证类。** 有真值的用程序，有真值又要看质量的用 `gated`，只有资料来源的用 `trajectory`：

```python
my_tasks = [
    {"id": "date", "instruction": "把 2024年3月5日 写成 ISO 8601 日期; 以「答案: 」开头给出",
     "verification": {"class": "exact", "gold": "2024-03-05"}},
    {"id": "sum-explained", "instruction": "12×12 等于多少, 并说明过程; 以「答案: 」开头给出结果",
     "verification": {"class": "gated",
                      "gate":  {"class": "exact", "gold": "144", "kind": "numeric"},
                      "score": {"class": "rubric", "criteria": [{"text": "说明了计算过程", "weight": 1}]}}},
    {"id": "summary", "instruction": "只依据下面的资料写一句总结。资料: X 公司 2023 年营收 412 亿元。",
     "observations": [{"tool_call_id": "t1", "tool": "doc", "observation": "X 公司 2023 年营收 412 亿元。"}],
     "verification": {"class": "trajectory", "grounding_policy": "must_ground"}},
]
```

**2. 接上模型和评委，把任务集变成配对流水线能吃的形态。** 适配器见 [examples/adapter_openai_compat.py](examples/adapter_openai_compat.py)（只用标准库，任何兼容端点）；`bench_tasks` 把每道题的 `evaluate` 结果按显式阈值二值化——"过"是 ≥ 多少要写进报告口径。

```python
from examples.adapter_openai_compat import make_call, make_llm   # 只用标准库的 OpenAI 兼容适配器

call_a, call_b = make_call(model="model-a"), make_call(model="model-b")
judge = make_llm(model="judge-model")
tasks = pe.bench_tasks(my_tasks, threshold=1.0, llm=judge)
```

**3. 三个对象，各固定各的变量。** `n` 是每题每系统的重复次数；交错运行轮转先后顺序。

```python
STRICT = "严格按要求输出, 不要任何多余内容。要求: "

def self_check(call):                     # agent 策略: 出草稿, 再按指令自检修正(多一次调用, 不加工具)
    return lambda p: call(f"指令: {p}\n草稿: {call(p)}\n检查草稿是否严格满足指令, 只输出修正后的答案。")

runs = {
    "model":   pe.run_interleaved({"A": pe.make_model(call_a), "B": pe.make_model(call_b)},
                                  tasks=tasks, n=6, prompt_prefix=STRICT),
    "harness": pe.run_interleaved({"strict": pe.make_model(lambda p: call_a(STRICT + p)),
                                   "bare": pe.make_model(call_a)}, tasks=tasks, n=6, prompt_prefix=""),
    "agent":   pe.run_interleaved({"single": pe.make_model(call_a),
                                   "self-check": pe.make_model(self_check(call_a))}, tasks=tasks, n=6, prompt_prefix=""),
}
for axis, run in runs.items():
    print(axis, pe.report(run["reports"], refusals=run["refusals"])["text"], sep="\n")
```

跑之前先算样本量，跑之后让 `interpret` 说结论：

```python
pe.required_tasks(0.30, 0.10)         # A 胜率 30%、负率 10%: 要多少配对任务才有 80% 功效
pe.required_pairs(0.15, 0.30)         # 连续分数: 均值差 0.15、sd 0.30 -> 内部真跑置换检验求 n(约 20 秒)
pe.detectable_effect(31)              # 反过来: 31 道题最小能检出多大的单方面胜率
pe.interpret(pe.paired_compare(scores_a, scores_b))["text"]   # 事后: 这个结果能说什么
```

`screen_tasks` / `screen_graded` 两阶段筛选帮你找有区分力的题——成功率 > 0.9 的近顶题默认排除，它们在任何可负担轮数下都筛不稳。

## 评一次真实的 agent 运行

agent 维度需要 agent 真实看到的工具观察，不是桩。[AgentXRay](https://github.com/alloevil/AgentXRay) 已经把 Claude Code / Codex / OpenClaw / Hermes / OMP / Gemini CLI 的日志规范成同一形状；`paired_eval.adapters.agentxray` 把这份导出变成 `trajectory` 任务，agent 最终回答里的每条 claim 都对着它实际看到的观察做 grounding：

```python
import json
from paired_eval.adapters import agentxray as ax

sess = json.load(open("tests/fixtures/agentxray-codex-session.json"))   # = curl http://localhost:3800/api/codex/sessions/<id>
task = ax.trajectory_task(sess)              # id, instruction, observations[], verification: trajectory
r = pe.evaluate(task, response=ax.final_answer(sess), observations=task["observations"], llm=judge)
print(r["score"])                            # 最终回答的 grounding 率
```

同一条指令在两个 agent（或两个 harness）下各跑一次，就是一个配对单元：各自打分，再交给 `pe.paired_compare`。

## 报告怎么读

报告里每个字段都在防一种误读——有效样本、触顶/触底、不一致对的集中度，以及四种结论
（显著 / 有界 null / 无信息 / 检验无力）。"p > 0.05" 有三种含义、三种处方：见
[docs/reading-the-report.md](docs/reading-the-report.md)。

与 lm-evaluation-harness、Inspect、promptfoo、openai/evals 的关系——它们跑评测，这里接住逐题分数并回答"这个差别可信吗"：
[docs/related-work.md](docs/related-work.md)。

## API 一览

`import paired_eval as pe`：

| 分组 | 入口 |
|---|---|
验证器 | `evaluate`（`exact` / `retrieval` / `trajectory` / `rubric` / `gated`）`evaluate_pair` `evaluate_batch` `validate_task` `grade_answer` `extract_claims` `verify_trajectory` `run_rubric` `rubric_canary` |
配对 A/B | `make_model` `bench_tasks` `judge_check` `run_interleaved` `run_paired` `run_repeated` `report` `pairwise_compare` `reliability_matrix` `saturation` |
统计 | `paired_compare` `mcnemar_exact` `holm_adjust` `wilson_ci` `pass_hat_k` `required_tasks` `required_pairs` `detectable_effect` `p_floor` `min_units_for_alpha` `interpret` |
筛题 | `screen_tasks` `screen_graded` · 内置 `ALL_TASKS`（31 道中文冒烟题，示例与自测用） |
适配 | `make_resilient` `throttled_pmap` `Meter` `set_language` · `paired_eval.adapters.agentxray` — AgentXRay 会话导出 → `trajectory` 任务（见下） |

## 范围与状态

- **是**：给模型 / agent / harness 做评测的方法与工具——可叠加的验证器（程序 gate + rubric score）、配对比较的诚实统计、样本量规划、筛题。
- **不是**：大规模题库（内置 31 题只作示例与自测）、评测平台、模型客户端、rubric **生成**器（autorubric 的那一半不在这里——这里评的是 rubric 会不会被骗）。跑大规模任务集请用上面那些框架，把逐题分数交给这里。
- **状态**：0.4.0，单作者，接口可能变。报告中英双语（`set_language`）；代码注释与内置题为中文。
- [docs/findings.md](docs/findings.md) 是用本工具对一对模型、三族任务做的案例研究：harness 与 agent 的效应各约 +0.6~0.75，模型效应 < 10%，两者收益递减但可叠加。数字是实例特定的，展示的是报告该怎么写。

## 什么时候用它

- 两个系统在**同一批题、同一顺序**上有逐题结果，而你要对外说清这个差别是否可信。
- 你在比模型与模型、脚手架与脚手架、agent 策略与 agent 策略，并且希望三者分开、各固定各的变量。
- 比较结果是负的，你需要报出**排除了多大的效应**，而不是一句"无差异"。
- 你怀疑题集已饱和（两系统恒过），或某个系统触顶 1.0 导致交互项失去意义。
- 你想在花模型调用之前先算样本量（`required_tasks` / `required_pairs`），或先知道现有题集根本能看见多小的差异（`detectable_effect`）。

## 什么时候不要用它

- **不是评测集**。内置 31 道中文冒烟题的 `detectable_effect(31)` 是 0.25，撑不起一个有把握的结论。请带自己的题，或用下面那些框架跑真实任务集、把逐题分数交过来。
- **不接受非配对结果**。同一批题、同一顺序是硬要求；只剩汇总分数时，配对无法事后补回来。
- **不是运行器、不是平台**。没有模型客户端、不跑任务、没有看板；也不**生成** rubric，`rubric_canary` 只检验一份 rubric 会不会被糊弄。
- **不产出普适结论**。一个题族上测到的效应不会自动迁移：本仓库为脚手架维度筛出的题，换比模型时就不敏感了。
- **接口尚未稳定**。0.4.0、单作者，接口可能变；代码注释与内置题是中文（报告中英双语）。

## 与 lm-evaluation-harness / Inspect / promptfoo / openai/evals 的关系

描述取自各项目自己的 README；完整表格见 [docs/related-work.md](docs/related-work.md)。

| 工具 | 它做什么 | 与本项目的关系 |
|---|---|---|
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 60+ 学术基准、多种模型后端；按指标报标准误 | 它跑评测；把逐题分数交给 `paired_compare` / `interpret` |
[Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) | 评测框架：提示工程、工具使用、多轮对话、模型评分；200+ 预置评测 | 同上；scorer 输出逐样本，天然可配对 |
[promptfoo](https://github.com/promptfoo/promptfoo) | 模型/提示 side-by-side 对比与断言；red teaming | 在"比较"上重叠；本项目补验证器叠加与统计结论层 |
[openai/evals](https://github.com/openai/evals) | 模板 + JSON 数据的评测注册表 | 同样的下游关系 |

## 常见问题

**怎么安装、需要什么？** 跑 `pip install paired-eval`。要求 Python ≥ 3.9，不装任何第三方包——`pyproject.toml` 里的空依赖列表是刻意的设计约束。模型与评委由你以两个可调用对象注入：`call(prompt) -> str` 与 `judge(prompt, system, schema) -> dict`；只用标准库的 OpenAI 兼容适配器随仓库提供，见 [examples/adapter_openai_compat.py](examples/adapter_openai_compat.py)。除此之外没有要配置的东西，`python3 -m paired_eval` 不接任何 API 就能跑一遍离线演示。

**A 在 40 道题上得 0.72、B 得 0.65，A 更好吗？** 只看这两个均值通常判断不了，这正是本库存在的理由。把两个系统的逐题结果按同一题序交给 `pe.paired_compare(a, b)`，再交给 `pe.interpret`：你会拿到效应量与 bootstrap 95% 区间、逐轮 McNemar 精确检验 p、逐题置换检验 p、多系统时的 Holm 校正、不一致对的方向与集中度，以及一句可直接写进报告的结论。如果两个系统跑的不是同一批题同一顺序，任何检验都补不回配对，只能重跑。

**评 model、评 harness、评 agent 有什么区别？** 这是三个问题，各自固定不同的变量，混着评什么都得不到。评 **model** 固定题集与脚手架、只换模型；评 **harness** 固定模型、换提示 / 脚手架 / 工具接线；评 **agent** 固定模型与脚手架、只换策略，例如单遍 vs 出草稿后自检修正。2×2 因子设计把两个因子放到同一把尺子上比主效应，并自动报出触顶/触底的格——参照格卡在 1.0 时算出的交互项是设计的产物，不是发现。

**为什么它不肯写"无显著差异"？** 因为这句话盖住了三种情况、三种处方。样本单元太少是**无信息**：这个设计本来就看不见现实量级的效应，处方是加单元。不一致对太少是**检验无力**：精确检验可达的最小 p 由不一致对数决定，效应再大也到不了显著，处方是加轮次或换能拉开差距的题。单元够、效应确实小，才是**有界的 null**——必须连同"排除了多大效应"一起报，例如"模型效应 < 10%"。这个纪律是本项目自己犯错后立的，过程记在 [docs/corrections.md](docs/corrections.md)，现已固化进 `interpret`，从 API 层面不可能再悄悄重犯。

**findings 里的数字可信吗？** 把它们当实例特定的用法示例，不要当可继承的结论：数据来自一对托管模型与几族任务，评自己的系统该复跑工具而不是抄这些数。在这个范围内它们有三重可审：最强的结论已变成 [`paired_eval/reproduce_findings.py`](paired_eval/reproduce_findings.py) 里的可执行断言，漂移会被发现而不是被继承；每一条被推翻或修正过的结论都留在 [docs/corrections.md](docs/corrections.md)，原文保留并指向推翻它的证据；每个对外发布的数字都带指标、方法、复现命令与证据链接，列在 [docs/claims.json](docs/claims.json)。

## 文档 · 贡献 · 许可

[文档索引](docs/README.md) · [方法学教训](docs/lessons.md)（每个统计原语防的是什么错）· [CONTRIBUTING.md](CONTRIBUTING.md) · MIT
