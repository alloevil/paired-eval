# 文档索引

| 文档 | 读者 | 内容 |
|---|---|---|
[findings.md](findings.md) | 想知道"这套工具测出了什么" | 三个维度（harness / model / agent）的实测终报与交互项，每条带统计量、有效样本、可排除范围与复现结果。**实例特定，不可继承。** |
[lessons.md](lessons.md) | 想把方法学用到自己的评测上 | 二十条教训，按"防的是什么错"组织：检验能不能说话、设计能不能回答问题、工具能不能被信任。代码注释里的锚点都指向这里。 |
[corrections.md](corrections.md) | 想知道这套结论可信到什么程度 | 被推翻或修正过的结论，原文保留、标注被什么推翻、指向新证据。 |

三份文档的关系：**findings 里每个数字都能追到 lessons 里一条纪律，corrections 里每条修正都指向 findings 里被改写的那一段。**

## 从哪里开始

- 第一次接触：先读根目录 [README](../README.md)（[English](../README.en.md)），再读 [lessons.md](lessons.md) 的第一节"检验能不能说话"。
- 要评自己的系统：README 的"快速开始"两段代码 + [`examples/adapter_openai_compat.py`](../examples/adapter_openai_compat.py)。
- 要贡献代码：[CONTRIBUTING.md](../CONTRIBUTING.md)。
- 要复现记录的结论：[`reproduce_findings.py`](../paired_eval/reproduce_findings.py) 头部注释；三项检查的成本见 [`checkall.sh`](../checkall.sh) 头部。

## 锚点约定

文档里的 `<a name="..."></a>` 是稳定锚点，代码注释以 `docs/<文件>.md#<锚点>` 引用。
`tests/test_runtests.py::test_docs_anchors_resolve` 保证每个引用都能解析——改名或删除锚点会让快速套件失败。
