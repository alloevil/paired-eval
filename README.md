# paired-eval

给模型、agent、harness 做**配对评测**的方法学工具箱——统计上诚实，且能自我验证。

[English](README.en.md) · [实测结论](docs/findings.md) · [方法学教训](docs/lessons.md) · [纠正记录](docs/corrections.md)

零第三方依赖，纯标准库 Python ≥ 3.9。所有模型调用都是**注入**的：本仓库不绑定任何供应商。

---

## 它回答什么问题

> "有真值就用程序验证，没有就用 rubric"——这个二分法方向对，但落地时需要三件事：
> **对象拆开、验证器变成谱系而非二分、rubric 自身也要被评测。**

本仓库是这三件事的可运行形态：

| 你要评的 | 控制变量 | 本仓库给的 |
|---|---|---|
**model**（能力上限） | harness 固定 | 交错重复 + pass@1 / pass^k + 有效样本诊断 |
**agent**（策略） | 环境固定 | 同上，外加轨迹忠实性（claim 逐条 grounding） |
**harness**（脚手架） | **模型固定**，A/B 配对 | McNemar + 置换检验 + Holm 校正 + 效应量 CI |
**三者的交互** | 2×2 因子设计 | 同一把尺子上比主效应，并自动报出天花板 |

验证器按"能用的最强层"选，层可叠加——**程序验证做 gate，rubric 做 score**：

```
exact 匹配 → 程序/执行验证 → 性质验证(round-trip / 不变量) → 检索核实 → judge / rubric
                                                                    ↑ 自身要过 canary(糊弄回答不得高分)
```

## 它最在意的一件事：让评测在不能说话时说"我不能说"

一次配对比较不显著，可能是三种完全不同的病，处方各异：

```python
>>> import claim_eval as ce
>>> c = ce.paired_compare(scores_a, scores_b)
>>> ce.interpret(c)["text"]
'检验无力: 非零差值对只有 3 个, 最小可能 p 是 0.250 >= alpha=0.05, 效应再大也不可能显著(...) —— 需至少 6 个非零差值对'
# 或: '未检出差异: p=0.727, 该设计只能排除 >=10% 的单方面胜率 (n=80, 点估计 Δ=+0.025); 低于此的效应无法排除'
# 或: '无信息的 null: n=0 太小, 任何效应都检不出(MDE 不可达) —— 不能当作『无差异』的证据'
```

这不是锦上添花。本仓库自己的实验里，**同一批数据被误读过三次**（把"样本不足"写成"效应不存在"、
把"检验到不了显著"读成"不显著"、用"CI 恰好排除 0"的公式当功效公式）——每一次都写进了
[docs/corrections.md](docs/corrections.md)，然后变成了 `interpret()` / `p_floor()` / `required_pairs()` 里的机制。

## 快速开始

```sh
sh runtests.sh            # 183 个测试, ~3 秒, 零外部依赖(pre-commit 用的就是它)
sh checkall.sh --fast     # 五层验证清单的前四层(~4 分钟); 不带 --fast 加跑变异棘轮(~6 分钟)
```

给两个系统做一次配对 A/B——只需提供 `call(prompt) -> str`：

```python
import paired_bench as pb

strict = pb.make_model(lambda p: my_llm("严格按要求输出,不要任何多余内容。要求: " + p))
bare   = pb.make_model(lambda p: my_llm(p))

tasks = [t for t in pb.ALL_TASKS if t["id"] in pb.SCAFFOLD_SENSITIVE]   # 对脚手架敏感的 4 题
out = pb.run_interleaved({"strict": strict, "bare": bare}, tasks=tasks, n=6, prompt_prefix="")
print(pb.report(out["reports"], refusals=out["refusals"])["text"])
```

`report()` 一次给全九项：效应量+CI、逐题置换 p、逐轮 McNemar、Holm 校正、不一致对分布与集中度、
有效样本、拒答归属、null 的可排除范围、触顶/触底诊断。**少一项，结论就可能失真**——这九项每一项都曾在手工报告里被漏掉过。

评一个带轨迹的 agent 回答是否忠实于它看到的观察：

```python
import eval_task as et
r = et.evaluate({"id": "q1", "instruction": "总结资料",
                 "verification": {"class": "trajectory", "grounding_policy": "must_ground"}},
                response=agent_answer, observations=tool_observations, llm=my_judge)
r["score"], r["metrics"]["fabricated"]    # grounding_rate, 以及无据 claim 条数
```

`llm(prompt, system, schema) -> dict` 是带 JSON-schema 约束的结构化调用；`examples/adapter_openai_compat.py`
给了一个只用标准库的 OpenAI 兼容适配器。

## 模块地图

| 文件 | 职责 |
|---|---|
`answer_match.py` | L1：数值/选项/集合/`\boxed{}` 的程序判定，全半角与单位归一 |
`claim_eval.py` | claim 抽取 → 检索/轨迹核实 → 三值判定；**统计原语**；成本计量、有界重试、限并发 |
`rubric_eval.py` | rubric 逐条二值判定 + 加权聚合 + canary（糊弄回答分离度） |
`eval_task.py` | 声明式 task schema，按 `exact / retrieval / trajectory / rubric` 路由；交错配对评判 |
`paired_bench.py` | 任务集、交错重复运行、可靠性透视、两两比较、筛题、完整报告 |
`reproduce_findings.py` | 把记录的结论变成可执行检查（需注入真实模型调用） |
`mutate_auto.py` / `mutate.sh` | 变异测试：AST 机械生成 + 手挑语义变异 + 等价变异基线棘轮 |
`runtests.sh` / `checkall.sh` / `check_hooks.py` | 快速套件、五层发布清单、钩子的真机验证 |

## 统计原语（`claim_eval`）

| 原语 | 回答的问题 |
|---|---|
`paired_compare(a, b)` | 配对差的置换 p、bootstrap CI、胜/负/平、**有效对数** |
`mcnemar_exact(a_only, b_only)` | 二值配对：不一致对的精确双侧 p |
`holm_adjust(pvalues)` | 多重比较控制 FWER |
`wilson_ci(k, n)` / `pass_hat_k(s, n, k)` | 比例区间 / 可靠性（k 次全过） |
`required_tasks(p_win, p_loss)` | 要多少配对任务才能检出这个胜率差 |
`required_pairs(mean_diff, sd)` | 连续分数版：内部真跑置换检验（规划器与检验器同一把尺子） |
`detectable_effect(n)` | 反问题：任务集固定时最小可检出效应（MDE） |
`p_floor(n)` / `min_units_for_alpha(a)` | 检验的 p 地板（2/2ⁿ）与所需最小单元数 |
`interpret(compare)` | 把结果翻译成可报告结论：显著 / 有界 null / 无信息 / 检验无力 |

`paired_bench.saturation()` 告诉你有多少题贡献了零信息；`screen_tasks()` / `screen_graded()` 用两阶段筛选找有余量的题——**近顶题（>0.9）在任何可负担轮数下都筛不稳**，默认直接排除。

## 它如何验证自己

```
1 快速套件      ~3s    183 测试 + 结构自检(定义数==执行数、__main__ 恰一个、文档锚点可解析、空套件即失败)
2 钩子集成      ~26s   建临时仓库 + 裸远端, 跑真实 pre-commit / pre-push
3 功效校准      ~5s    required_pairs 与教科书功效公式对量级(真跑蒙特卡洛)
4 手挑变异      ~81s   42 个高阶语义变异 + 腐坏检测
5 机械变异棘轮  ~6min  400+ 变异点全量, 与已归档等价变异基线比对, 新增存活即失败
6 结论复现      手动   reproduce_findings.py: 三条记录结论的方向/量级/显著性/有效样本(需真实模型)
```

变异测试是这里的核心纪律：**测试通过不等于测试有效**。已归档 30 条经分析确认的等价变异——为它们补测试只会把实现细节钉死。
每一层都曾在建立过程中抓到至少一个真实缺陷（含一个"181 个单测全绿时集成测试第一次运行就发现"的诊断错误）。

## 关于 docs/ 里的数字

[findings.md](docs/findings.md) 里的结论（脚手架 +0.625、模型 <10%、自检增量 +0.183…）是关于**一对特定模型和几族特定任务**的测量。
它们展示工具怎么用、结论该怎么写（附界、附集中度、附能排除多大效应），**不是可继承的普适结论**。
拿本仓库评你自己的系统时，该复跑的是工具，不是抄这些数。

[lessons.md](docs/lessons.md) 是通用的：每条都是在真实数据上先被骗过一次、然后变成代码里的机制。
代码注释里指向 `docs/lessons.md` 的锚点都落在这里。

## 设计原则

- **依赖注入，零供应商绑定**：`llm` / `search` / `call` / `judge` 全部由调用方提供；测试用确定性假对象跑完全部逻辑。
- **规划器与检验器用同一把尺子**：样本量规划内部跑的就是实际使用的检验，不用闭式近似。
- **null 必须附界**：不许写"无差异"，只许写"排除了 ≥X% 的效应"或"检验无力，需 N 个 ×××"。
- **门禁必须自测**：每一层验证都有下一层验证它。
- **等价变异记录而非补测**；**纠正留痕而非覆盖**。

## 许可

MIT。代码与文档为中文；English README 见 [README.en.md](README.en.md)。
