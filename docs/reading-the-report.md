# Reading the report · 报告怎么读

`report()` / `interpret()` 的每个字段防的是一种误读。英文在前，中文在后。

## Fields

| Field | Meaning |
|---|---|
informative sample | Tasks both systems always pass or always fail carry no information; read the MDE against the informative count |
refusals | Dropped calls per system — the refusal rate is part of the result |
at ceiling / at floor | A system at 1.0 or 0.0 has no headroom; effects against it are compressed, interactions uninterpretable |
Δ, CI95 | Effect size with a bootstrap 95% interval |
per-task p / per-round McNemar / Holm | The two paired tests; Holm correction when several systems are compared |
discordant a:b, concentration | Direction and spread of the disagreements; 1.0 = all from one task |
verdict | `significant` / `null` — the four-way distinction is in the report text and in `p_floor` / `rules_out`: bounded null (with the ruled-out effect) · uninformative · powerless (with what is missing) |

"p > 0.05" means three different things with different remedies: too few units is *uninformative*; too few discordant pairs is *powerless* (add rounds, or tasks that separate the systems); enough units and still nothing is a *bounded null* — report the effect it rules out, never "no difference".


## 字段

| 字段 | 含义 |
|---|---|
有效样本 | 两系统全轮次同结果的题不贡献信息；MDE 按有信息题数读 |
拒答 | 各系统被丢弃的调用数——拒答率是结果的一部分 |
触顶 / 触底 | 满分或零分的系统无余量；与它比较效应被压缩，交互项不可解释 |
Δ, CI95 | 效应量与 bootstrap 95% 区间 |
逐题 p / 逐轮 McNemar / Holm | 两种配对检验；多系统时的多重校正 |
不一致对 a:b, 集中度 | 分歧的方向与分布；1.0 = 全部来自一道题 |
结论句 | 显著 · 有界 null（附能排除多大效应）· 无信息 · 检验无力（附缺多少什么） |

"p > 0.05" 有三种含义，处方各异：样本太小是无信息；不一致对不够是检验无力（加轮次或换能拉开差距的题）；样本够却没测出来才是有界的 null——报能排除多大效应，而不是"无差异"。

