## 为什么 / Why

<!-- diff 已经说了做了什么; 这里写为什么。改结论请指向推翻它的证据。 -->
<!-- The diff says what; say why. If a conclusion changes, point at the evidence that overturns it. -->

## 验证 / Verification

- [ ] `sh runtests.sh` 全绿（若本机有多个 Python，`sh checkall.sh --fast` 会用每个都跑一遍）
- [ ] 改了库代码：`python3 mutate_auto.py <模块> --check` 无新增存活，或等价变异已在 `mutate_auto.py` 头部写明理由并刷新基线
- [ ] 改了行为：对应的 `docs/` 已更新；锚点自检通过
- [ ] 改了 README 示例：`tests/test_readme.py` 仍通过（示例是被真跑的）
- [ ] 推翻了既有结论：原文保留在 `docs/corrections.md`，标注被什么推翻
