# 参与贡献 / Contributing

中文为主；English summary at the end.

## 一分钟上手

```sh
git clone git@github.com:alloevil/paired-eval.git && cd paired-eval
git config core.hooksPath hooks     # 启用仓库自带的 pre-commit(3s 快速套件) 与 pre-push(改动行变异检查, ~30s)
sh runtests.sh                      # 192 测试, ~3 秒, 零外部依赖
```

零第三方依赖是**设计约束**，不是暂时状态：新增 `import` 只能来自标准库。

## 改动前要知道的三件事

**1. 验证是分层的，每层有它的理由。** `sh checkall.sh --fast` 跑前四层（本地 ~4 分钟、CI ~8 分钟），
`sh checkall.sh` 加跑第五层机械变异棘轮（~25 分钟）。每层为什么存在、为什么放在那一层，写在 [`checkall.sh`](checkall.sh) 头部。
本机装了多个 Python 时，第 1b 层会用每个版本重跑快速套件——**只在一个解释器版本上绿过，就还没绿过**
（[为什么](docs/lessons.md#float-equality)）。

**2. 新代码要过变异棘轮。** 推送时 pre-push 会对改动行做机械变异：新增代码若有变异体能在测试全绿下存活，推送被拒。
处理只有两种，不要绕：
- 是真缺口 → 补一个能杀死它的测试（用[探针法](docs/lessons.md#probe-classification)定位该断言什么）；
- 是等价变异（任何输入都无法区分）→ 在 [`mutate_auto.py`](mutate_auto.py) 头部的等价类清单里**写明理由**，然后
  `python3 mutate_auto.py <模块> --baseline` 刷新基线。为等价变异写测试只会把实现细节钉死，
  让合理重构无谓失败（[为什么](docs/lessons.md#mutation)）。

**3. 统计代码有额外的纪律。** 改 `claim_eval.py` 的统计原语或 `paired_bench.report()` 前读 [lessons.md](docs/lessons.md) 第一节：
null 必须附界、p 地板的基数不是单元数、规划器与检验器用同一把尺子。蒙特卡洛函数要能注入确定性随机源，边界才能被精确断言。

## 提交与 PR

- **提交信息写"为什么"，不写"做了什么"**——diff 已经说了做了什么。本仓库的提交历史是它的第二份文档。
- 一个 PR 一件事。改了行为就改对应的 docs；改了 docs 里的锚点，`test_docs_anchors_resolve` 会告诉你哪里悬空了。
- 推翻既有结论时**不要覆盖原文**：保留、标注被什么推翻、指向新证据（见 [corrections.md](docs/corrections.md) 的写法）。
- README 里的示例代码是被测试真跑的（`tests/test_readme.py`），改示例就是改代码。

## 报告问题

- 方法学疑问（"这个统计量为什么这样算"）与 bug 一样欢迎——用 issue 模板里的"方法学问题"。
- 发现结论漂移（`reproduce_findings.py` 报 FAIL）：附上完整输出与你用的模型；**方向反转要当发现处理，不要当失败**。

---

## English summary

- Zero third-party dependencies is a design constraint; only the standard library may be imported.
- Enable the bundled hooks (`git config core.hooksPath hooks`); `sh runtests.sh` is the 3-second fast suite, `sh checkall.sh --fast` the four cheap verification layers, `sh checkall.sh` adds the ~25-minute mutation ratchet.
- New code must survive the ratchet: a surviving mutant is either a real test gap (write the killing test) or an equivalent mutant (document *why* in `mutate_auto.py`'s header, then refresh the baseline with `--baseline`). Never write a test whose only purpose is to kill an equivalent mutant.
- Statistical code has extra discipline — read the first section of [docs/lessons.md](docs/lessons.md) before touching `claim_eval.py` primitives or `paired_bench.report()`.
- Commit messages explain *why*; corrections are recorded, not overwritten; README examples are executed by tests.
