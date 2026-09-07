# How it relates to other tools · 与其他工具的关系

## English

They are frameworks for *running* evaluations; paired-eval sits downstream and does not duplicate task libraries or model backends. Descriptions are taken from each project's own README.

| Tool | What it does | Relation |
|---|---|---|
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 60+ academic benchmarks over many model backends; per-metric standard errors | Produces per-task scores → feed them to `paired_compare` / `interpret` |
[Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) | Eval framework: prompt engineering, tool use, multi-turn dialog, model-graded components; 200+ pre-built evals | Same; scorer output is per-sample and pairs naturally |
[promptfoo](https://github.com/promptfoo/promptfoo) | Side-by-side model/prompt comparison with assertions; red teaming | Overlaps on "compare"; paired-eval adds stacked verifiers and the statistical verdict layer |
[openai/evals](https://github.com/openai/evals) | Registry of template-based evals fed by JSON data | Same downstream relation |


## 中文

它们是*运行*评测的框架；本项目接在下游，不重复任务库与模型后端。描述取自各项目自己的 README。

| 工具 | 它做什么 | 关系 |
|---|---|---|
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 60+ 学术基准、多种模型后端；按指标报标准误 | 它产出逐题分数 → 交给 `paired_compare` / `interpret` |
[Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) | 评测框架：提示工程、工具使用、多轮对话、模型评分；200+ 预置评测 | 同上；scorer 输出逐样本，天然可配对 |
[promptfoo](https://github.com/promptfoo/promptfoo) | 模型/提示 side-by-side 对比与断言；red teaming | 在"比较"上重叠；本项目补验证器叠加与统计结论层 |
[openai/evals](https://github.com/openai/evals) | 模板 + JSON 数据的评测注册表 | 同样的下游关系 |

