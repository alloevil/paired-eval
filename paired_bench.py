# -*- coding: utf-8 -*-
"""paired_bench — 本仓库自带的配对 A/B 冒烟任务集与工作流(实战沉淀,全程序判定)。

任务 schema: {"id", "instruction", "check": str->bool, "canonical": 一个合法输出样例}
model 依赖注入: model(prompt: str) -> str | None (None = 拒答/不可用 -> 该题成对丢弃)。

入口:
    make_model(call)                          # 裸调用适配: 异常有界重试, 耗尽转 None
    run_paired(a, b, tasks)                   # 单发配对: 逐题交替先后 -> 判定 -> 置换检验
    run_repeated(model, tasks, n=8)           # 单系统复核: pass@1(能力) vs pass^k(可靠性)
    run_paired_repeated(a, b, tasks, n=8)     # 两系统交错重复(ABBA) + 逐轮不一致对/McNemar
    run_interleaved({name: model}, tasks, n)  # N系统交错重复(顺序轮转) -> 喂 reliability_matrix
    reliability_matrix(reports)               # 跨系统逐题透视 + 分歧标记(三道守卫)
    pairwise_compare(reports)                 # N系统两两比较 + Holm 多重比较校正
    screen_tasks(candidates, models)          # 新题入库前两阶段信息量筛选(筛+复核)
    saturation(reports)                       # 有效样本诊断: 饱和题贡献零信息
门禁 fixture(开箱即用, 没有 fixture 的门禁不会被跑):
    TRAJECTORY_GATE -> claim_eval.trajectory_selfcheck(pb.TRAJECTORY_GATE, llm)
    WORLD_GATE      -> claim_eval.selfcheck(pb.WORLD_GATE, llm, search)  # 依赖实时检索,会腐坏
    RUBRIC_GATE     -> rubric_eval.rubric_canary(g["criteria"], g["good"], g["fooling"], llm)
判定器故意严格(strip 后精确比较): 测的就是指令遵循,宽松即失真。
纪律: 单发分歧项必须经 run_repeated 复核才许下结论(有误标前科);
多系统比较必须走交错路径, 逐系统分别 run_repeated 再并排会被 reliability_matrix 拒绝。
最近一次发布态真机验证(全路由 + 近15轮新增字段):
    exact      reason=no_slot 正确归因, 批次 non_attempt_reasons 汇总正常
    retrieval   观点句进 unverifiable_claims, queries 逐条留痕且中性(无声称值)
    trajectory  must_ground 0.67 vs allow_parametric 1.00(parametric=1), 加权>未加权
                说明那条无据补充是 detail 而非 core —— 重大幻觉不被无害细节稀释
    bench       orders 逐轮交替, refusals/attempts 齐备; RUBRIC_GATE 分离度 1.00
harness 评测(固定模型比脚手架)与模型评测需要不同的题: 后者要能力梯度, 前者要
对输出格式敏感的题。实测(default 固定, 严格脚手架 vs 裸指令, 交错重复):
    if-json    strict 6/6(pass^3=1.00) vs bare 2/6(pass^3=0.00), 复核一致
               机理: 裸指令下模型输出 markdown 围栏 ```json ... ```,
               JSON 本身正确但 json.loads 失败 —— 脚手架缺失导致的输出污染,
               不是能力问题。这是 harness 差异最典型的形态。
    if-3lines  10/10 vs 8/10 首轮有差, 复核 6/6 vs 6/6 无差 -> 弱信号, 不足为据
    其余6题     两侧恒过, 零区分力 —— saturation() 报 2/8 有信息, 加题无用
拓宽子集后的终报(4 题敏感子集, n=6, 交错+轮转):
    不一致对 0:13 全部偏向严格脚手架, McNemar p=0.000244(Holm 后不变)
    Δ=-0.542 CI95=[-0.833,-0.250], 有效样本 4/4 题
    集中度 0.46 —— 13 个不一致对分布在全部 4 题(2/6/1/4), 无单题贡献过半
结论: "严格脚手架对结构化输出是必需的" —— 效应在 JSON对象/JSON数组/key=value/YAML
四种独立格式上一致复现, 故不是单题异常; 但 5 道单值输出题两侧恒过, 故结论不外推到
"脚手架普遍更好"。拓宽前(仅 if-json)集中度 1.00, 只能支撑"某一题上有差异"。
交叉验证(用同样 4 题改比模型, 脚手架固定=严格): 有效样本仅 1/4, 不一致对 3:1,
p=0.625 —— 两个模型在严格脚手架下都能正确产出结构化格式(仅 YAML 上 5/6 vs 3/6)。
该设计 24 个配对单元, MDE=0.32: 这条 null 只能排除 >=32% 的效应(界较弱);
更紧的界见第107轮的 80 单元重测(MDE=0.10, 模型效应<10%)。
即"任务的敏感轴是特异的": 为脚手架维度筛出的题不能自动用于模型维度, 两个维度
必须各自筛选(这也解释了第 57 轮为模型维度找区分题时 12/12 全饱和的挫败)。
SCAFFOLD_SENSITIVE 见文件末尾。
模型维度的诚实披露: 本 bench 无法区分 smol 与 default。四次独立筛选共 36 道候选题
全部两侧恒过(saturation 报有信息题为 0):
    第57轮  12 道单步硬题(10字符倒序、1234×5678、第20个素数)      12/12 恒过
    第58轮  10 道多约束组合题(首字母递增、7字含约束、回文、矩阵)   9/10 恒过, 1 道复核后为噪声
    第101轮  8 道格式脆弱题 -> 3 道对脚手架敏感, 但交叉验证对模型不敏感
    第104轮  8 道多步计算题(多步算术、排序取位、链式串操作、日期改写) 8/8 恒过
饱和这种 null 的强度可直接算: 36 候选 x 2 轮 = 72 个配对单元零不一致对, 若真实
单方面胜率为 10%, 出现这种全零的概率仅 5.1e-4(20% 时 1.1e-7) —— 故这条 null 很强,
与上面那些 4 单元的弱 null 不同(见各处 MDE 标注)。
结论: 在单轮、程序可判定的任务上, 这两个模型确实等价 —— 不是造题能力不足(四种
难度轴都试过), 而是差异不在此类任务上。要区分它们需换任务族: 长程多轮、工具使用、
需要外部知识的推理。这类任务超出本 bench 定位(冒烟集), 应在宿主系统里建。
忠实性轴上的模型 A/B(第105轮, 换任务族的尝试: 同观察同judge, 比两模型的无据率):
    条件"仅依据资料"      两模型 8/8 claim 全 grounded, 无据率 0%, Δ=0.000 p=1.000
    条件"补充相关背景"    default 84/98 无据(86%) vs smol 97/109 无据(89%)
                          模型间 Δ=+0.009 p=0.500(2:2) —— 仍不可区分
    提示条件的效应: 0% -> 87%, 决定性
    界的说明: 忠实性 A/B 只有 4 个配对提示, MDE 在该样本量下不可达 —— 两条"模型
    不可区分"是无信息的 null, 不能当证据用; 模型维度的可用界只有第107轮那条(<10%)。
    但"提示条件效应"这一侧是 206 条 claim 上的 0% vs 87%, 与样本量无关地成立。
结论: 忠实性主要由提示约束决定, 而非模型选择 —— 与"结构化输出由脚手架决定"同源。
两次独立实验都指向同一点: 在这两个模型之间, harness 效应量远大于 model 效应量。
副产品: 验证了 must_ground 政策的判别力(score 1.00 -> 0.07~0.11)与 fabricated 计数的敏感性。
结论的可复现性(第111轮): reproduce_findings.py 把上面最强的那条发现变成可执行断言
  (方向/量级/显著性/有效样本四项判据, 阈值宽于记录值以容纳正常噪声)。
  首次复跑: Δ=0.688(记录 0.625), Holm 后 p=0.00098(记录 0.0117, 更强), 方向一致,
  有效样本 3/4, 集中度 0.36(差异跨题复现而非单题异常) -> PASS。
  为什么必要: 第44轮见过同模型同题跨时间窗 0/8 vs 8/8 的漂移。记录在文档里的数字
  是历史声明; 只有可复跑的检查才能让漂移被发现而不是被继承。判据本身由
  test_reproduce_findings.py 用假 call 覆盖(方向反转要报"重大发现"而非失败, 饱和的
  处方是换题而非放宽阈值) —— 与第75轮"门禁必须自测"同源。
null 结论的复现(第112轮): check_model_null 把"模型不可区分"也变成可执行检查, 且区分
  两件不同的事 —— problems="差异变得可检出"(真漂移, 模型可能被更新), warnings="null
  仍成立但界更松"(不是失败, 但不构成复现)。真实复跑: Δ=-0.125, p=0.500 -> null 成立,
  但 16 单元只能排除 >=46%, 故如实报 WARN 并给出补齐所需样本量(81 单元)。
  这条分界正是第107轮我犯错的地方: 样本不足的 null 会伪装成"复现成功"。

agent 轴终报(第113轮, 补上三维中最弱的一环): 同模型、同裸指令, 只换策略 ——
  单遍 vs "出草稿后按指令自检修正"(多一次调用, 不加任何工具或提示约束)。
    Δ=+0.750 CI95=[+0.250,+1.000] 逐轮 McNemar p=0.00049 不一致对 12:0 集中度 0.33
  即策略效应与脚手架效应同量级(+0.750 vs +0.625~0.688), 且方向一致、跨题复现。

harness x agent 的 2x2(同一批题、同一次交错运行, 3 轮):
                single   check
    bare         0.333   1.000
    strict       1.000   1.000
  harness 主效应 +0.333 | agent 主效应 +0.333 | 交互项 -0.667
  六对比较(Holm 校正): 三对含 bare-single 的全部显著(p=0.047), 另三对全部 Δ=0.000。
  结论: 两个因子是替代品而非互补品 —— 它们修的是同一批失败(裸指令下模型加解释性
  文字), 任一到位即够, 两者都做是浪费。这是本仓库唯一一次测出强负交互, 也解释了
  为什么第106轮的 harness 主效应那么大: 它填的是同一个坑。
  自我纠正(第114轮): 上面的交互项 -0.667 有天花板成分, 必须拆成两半读 ——
    可测的一半(结论成立): bare-check 达到 1.000, 与 strict-single 在 12 个单元上
      零不一致对完全并驾。若自检只修部分失败, bare-check 会落在 strict-single 之下。
      故"自检单独就补齐了脚手架的全部收益"是有据的。
    不可测的一半(不能声称): strict-check 是否优于 strict-single —— 后者已在 1.000,
      无余量可加, 该格差异恒为 0 是设计所致而非发现。
    补救尝试失败: 另造 6 道更硬的题(JSON键序+转义、词长阶梯、YAML质数、隔位大写反转、
      恰好15字符、字母计数降序)在严格脚手架下全部 2/2 —— 第五次筛不出余量, 与第104轮
      的模型维度披露同因: 本任务族对该模型太易。要测这半个交互需换任务族。
  实践含义(据可测的那一半): "换脚手架"与"加自检"在此任务族上收益等同, 是二选一的
  成本决策(自检要多一次模型调用, 脚手架只多一段前缀 -> 同等收益下选脚手架);
  但"两者叠加无额外收益"未经证实, 不要据此拒绝叠加。
机械化(第114轮): report() 现在自动报出触顶/触底系统, 并声明"这些系统之间的 null 与
  任何交互项都不可解释"。上轮的报告没提天花板 —— 靠人记得, 就会有忘的时候。

2x2 因子设计终报(第106轮, 把两个因子放到同一把尺子上): 4 道敏感题 x 4 轮, 交错+轮转
              strict        bare
    default   14/16 .875   4/16 .250
    smol      14/16 .875   4/16 .250
    脚手架主效应 +0.625  Holm后 p=0.0117 (4对全显著, 不一致对 0:10)
    模型主效应   +0.000  Holm后 p=1.000  (2对全 1:1 / 2:2, 纯噪声对称)
    交互项       +0.000  -> 脚手架收益对两模型完全相同,
                           "小模型更依赖脚手架"这一常见直觉在此不成立
自我纠正(第107轮): 上面写的"model 效应根本不存在"是过度声称 —— 16 个配对单元的
MDE 是 0.46, 即那个 null 只能排除 >=46% 的效应, 是很弱的界。收紧后重测:
    4 题 x 20 轮 = 80 单元, MDE=0.10: Δ=+0.025 CI95=[0.000,+0.075],
    不一致对 5:3, p=0.727, 差异全集中在 fmt-yaml(20/20 vs 20/20 三题相同, yaml 15/20 vs 13/20)
可报告结论应为: 模型效应 < 10%(点估计 2.5%), 脚手架效应 +62.5% —— 相差至少 6 倍。
教训: null 结论必须附"能排除多大效应", 否则 "Δ=0.000, p=1.000" 会被读成"已证明为零"。
这是本仓库最有力的单一测量: 同任务、同单元、同统计机制, 两个因子直接可比。
功效上限(诚实披露): ALL_TASKS 共 31 题, claim_eval.detectable_effect(31)≈0.24 ——
即"较优系统需在 ≥24% 的题上单方面胜出且几乎无反向失误"才可能显著。这是冒烟集,
不是能定论的评测集; 要检出 10% 量级的差异需上百道配对任务(见 required_tasks)。
"""

import time

import answer_match as am
import claim_eval as ce


def make_model(call, tries=2, sleep=time.sleep):
    """把会抛异常的裸调用 call(prompt)->str 适配成 bench 契约 model(prompt)->str|None:
    异常有界重试,耗尽返回 None(该题成对丢弃,不崩整批)。
    全部任务被丢弃时 run_paired 会报错 —— 持续性故障不会被静默吞成"无数据"。"""
    def model(prompt):
        for i in range(tries):
            try:
                return str(call(prompt))
            except Exception:
                if i < tries - 1:
                    sleep(2.0 * (i + 1))
        return None
    return model


def _num(gold, tol):
    return lambda r: am.match_numeric(r, gold, rel_tol=tol)


def _eq(s):
    return lambda r: r.strip() == s


EXACT_QA = [
    {"id": "qa-everest", "instruction": "珠穆朗玛峰海拔约多少米? 一行作答,格式: 答案: <数字>",
     "check": _num("8848", 0.01), "canonical": "答案: 8848"},
    {"id": "qa-lightspeed", "instruction": "光在真空中的传播速度约为每秒多少公里? 一行作答,格式: 答案: <数字>",
     "check": _num("300000", 0.01), "canonical": "答案: 300000"},
    {"id": "qa-boil", "instruction": "标准大气压下水的沸点是摄氏多少度? 一行作答,格式: 答案: <数字>",
     "check": _num("100", 0.01), "canonical": "答案: 100"},
    {"id": "qa-17x23", "instruction": "17乘以23等于多少? 一行作答,格式: 答案: <数字>",
     "check": _num("391", 0.001), "canonical": "答案: 391"},
    {"id": "qa-2p10", "instruction": "2的10次方等于多少? 一行作答,格式: 答案: <数字>",
     "check": _num("1024", 0.001), "canonical": "答案: 1024"},
    {"id": "qa-week", "instruction": "一周共有多少小时? 一行作答,格式: 答案: <数字>",
     "check": _num("168", 0.001), "canonical": "答案: 168"},
    {"id": "qa-fib15", "instruction": "斐波那契数列(1,1,2,3,...)的第15项是多少? 一行作答,格式: 答案: <数字>",
     "check": _num("610", 0.001), "canonical": "答案: 610"},
    {"id": "qa-847x362", "instruction": "计算 847 × 362。一行作答,格式: 答案: <数字>",
     "check": _num("306614", 0.001), "canonical": "答案: 306614"},
]

IF_TASKS = [
    {"id": "if-3lines", "instruction": "用恰好三行列出三种水果,每行只写一个词,不要编号不要标点",
     "check": lambda r: (lambda ls: len(ls) == 3 and all(l.strip() and len(l.strip()) <= 4 for l in ls))(r.strip().split("\n")),
     "canonical": "苹果\n香蕉\n梨"},
    {"id": "if-json", "instruction": "输出一个合法的JSON对象,恰好含两个键a和b,值都是整数",
     "check": lambda r: (lambda d: isinstance(d, dict) and set(d) == {"a", "b"}
                         and all(isinstance(v, int) and not isinstance(v, bool) for v in d.values()))(__import__("json").loads(r.strip())),
     "canonical": '{"a": 1, "b": 2}'},
    {"id": "if-nomoon", "instruction": "用一句话介绍月亮,句中必须包含'卫星'一词,且全句不得出现'月'这个字",
     "check": lambda r: "卫星" in r and "月" not in r and len(r.strip()) >= 5,
     "canonical": "地球唯一的天然卫星,夜空中最亮的天体之一。"},
    {"id": "if-evens", "instruction": "输出1到10之间的所有偶数,用英文逗号分隔,不含任何空格",
     "check": _eq("2,4,6,8,10"), "canonical": "2,4,6,8,10"},
    {"id": "if-upper", "instruction": "将 hello world 转为全大写输出,只输出结果",
     "check": _eq("HELLO WORLD"), "canonical": "HELLO WORLD"},
    {"id": "if-pi8", "instruction": "只输出圆周率的前8位数字(从3开始,不含小数点)",
     "check": _eq("31415926"), "canonical": "31415926"},
    {"id": "if-prime1", "instruction": "回答下面的问题,只允许输出'是'或'否'一个字: 1是质数吗?",
     "check": _eq("否"), "canonical": "否"},
    {"id": "if-abc4", "instruction": "输出字符串 abc 重复4次拼接的结果,只输出结果",
     "check": _eq("abcabcabcabc"), "canonical": "abcabcabcabc"},
]

CHAR_TASKS = [
    {"id": "ch-rev4", "instruction": "倒序输出'人工智能'四个字,只输出结果",
     "check": _eq("能智工人"), "canonical": "能智工人"},
    {"id": "ch-rev6", "instruction": "倒序输出'机器学习模型'六个字,只输出结果",
     "check": _eq("型模习学器机"), "canonical": "型模习学器机"},
    {"id": "ch-revabc", "instruction": "倒序输出字符串 abcdef,只输出结果",
     "check": _eq("fedcba"), "canonical": "fedcba"},
    {"id": "ch-rev10203", "instruction": "倒序输出 10203,只输出结果",
     "check": _eq("30201"), "canonical": "30201"},
    {"id": "ch-strawberry", "instruction": "单词 strawberry 中有几个字母 r? 只输出数字",
     "check": _eq("3"), "canonical": "3"},
    {"id": "ch-evaluation", "instruction": "单词 evaluation 中有几个字母 a? 只输出数字",
     "check": _eq("2"), "canonical": "2"},
    {"id": "ch-mississippi", "instruction": "单词 mississippi 中有几个字母 s? 只输出数字",
     "check": _eq("4"), "canonical": "4"},
    {"id": "ch-third", "instruction": "'监督学习'的第三个字是什么? 只输出那个字",
     "check": _eq("学"), "canonical": "学"},
    {"id": "ch-fourth", "instruction": "'接口协议规范'的第四个字是什么? 只输出那个字",
     "check": _eq("议"), "canonical": "议"},
    {"id": "ch-tf5", "instruction": "单词 transformer 的第5个字母是什么? 只输出那个字母",
     "check": lambda r: r.strip().lower() == "s", "canonical": "s"},
    {"id": "ch-del", "instruction": "把'评测系统'中的'测'字删掉后输出,只输出结果",
     "check": _eq("评系统"), "canonical": "评系统"},
    {"id": "ch-revq", "instruction": "倒序输出'参数量化',只输出结果",
     "check": _eq("化量数参"), "canonical": "化量数参"},
]


# 格式脆弱题(经两阶段筛选+复核确认对脚手架敏感, 见 SCAFFOLD_SENSITIVE 注释)。
# 共同失败形态: 裸指令下模型加 markdown 围栏或前置解释, 内容对而格式坏。
FORMAT_TASKS = [
    {"id": "fmt-jsonarr", "instruction": "输出一个JSON数组,恰好包含3个整数元素",
     "check": lambda r: (lambda d: isinstance(d, list) and len(d) == 3
                         and all(isinstance(x, int) and not isinstance(x, bool) for x in d))(
                         __import__("json").loads(r.strip())),
     "canonical": "[1, 2, 3]"},
    {"id": "fmt-kv", "instruction": "输出两行,每行形如 key=value(无空格),key 分别是 name 和 age",
     "check": lambda r: (lambda ls: len(ls) == 2 and ls[0].startswith("name=")
                         and ls[1].startswith("age=") and all("=" in l and " " not in l for l in ls))(
                         [l for l in r.strip().split("\n") if l.strip()]),
     "canonical": "name=alice\nage=30"},
    {"id": "fmt-yaml", "instruction": "输出两行YAML,分别是 a: 1 和 b: 2(冒号后一个空格)",
     "check": lambda r: [l.rstrip() for l in r.strip().split("\n") if l.strip()] == ["a: 1", "b: 2"],
     "canonical": "a: 1\nb: 2"},
]

ALL_TASKS = EXACT_QA + IF_TASKS + CHAR_TASKS + FORMAT_TASKS


_TRAIL_CAP = 120  # 留痕截断长度: 够诊断失败模式,不撑爆报告


def run_paired(model_a, model_b, tasks=ALL_TASKS, prompt_prefix="严格按要求输出,不要任何多余内容。要求: "):
    """完整配对 A/B: 双侧作答 -> 程序判定 -> 任一侧 None 成对丢弃 -> 置换检验。
    调用顺序逐题交替(偶数题先A,奇数题先B): 固定先A后B会把窗口内漂移与预热效应
    系统性地压在同一侧 —— 与 judge 评测里交换位置消除 position bias 同理。
    返回 {"rows": [{"id","a","b","resp_a","resp_b","a_first"}], "dropped": [id],
          "dropped_detail": [{"id","sides"}], "compare": ...}。
    原始输出留痕(截断): 分歧项诊断靠看失败输出长什么样,只留 0/1 无法审计。"""
    rows, dropped, dropped_detail = [], [], []
    for idx, t in enumerate(tasks):
        prompt = prompt_prefix + t["instruction"]
        a_first = idx % 2 == 0
        if a_first:
            ra = model_a(prompt)
            rb = model_b(prompt)
        else:
            rb = model_b(prompt)
            ra = model_a(prompt)
        if ra is None or rb is None:
            dropped.append(t["id"])
            # 哪一侧拒答是系统属性(安全过滤/不可用), A侧全拒与双侧偶发不是一回事
            dropped_detail.append({"id": t["id"],
                                   "sides": [s for s, r in (("a", ra), ("b", rb)) if r is None]})
            continue

        def score(resp):
            try:
                return 1.0 if t["check"](str(resp)) else 0.0
            except Exception:
                return 0.0   # 输出连判定器都解析不了 = 不合规
        rows.append({"id": t["id"], "a": score(ra), "b": score(rb), "a_first": a_first,
                     "resp_a": str(ra)[:_TRAIL_CAP], "resp_b": str(rb)[:_TRAIL_CAP],
                     "measured_at": time.time()})
    if not rows:
        raise ValueError("全部任务被成对丢弃,无可比数据")
    compare = ce.paired_compare([r["a"] for r in rows], [r["b"] for r in rows])
    refusals = {"a": 0, "b": 0}
    for d in dropped_detail:
        for s in d["sides"]:
            refusals[s] += 1
    attempts = len(tasks)      # 单发: 每题每侧一次调用
    return {"rows": rows, "dropped": dropped, "dropped_detail": dropped_detail,
            "refusals": refusals, "attempts_per_side": attempts,
            "refusal_rate": {s: c / attempts if attempts else None
                             for s, c in refusals.items()},
            "compare": compare}


def run_repeated(model, tasks=ALL_TASKS, n=8, k=3,
                 prompt_prefix="严格按要求输出,不要任何多余内容。要求: "):
    """单模型逐题重复 n 次: 把能力(pass@1)与可靠性(pass^k)剥开。
    单发配对会把"抛硬币"误标成"能力缺口"(本仓库有真实案例: 某项连败2次,
    n=8 复核实为 4/8) —— run_paired 发现的分歧项必须用本函数复核后才许下结论。
    跨会话漂移警告: 本仓库实测同一模型同一题 n=8 得 0/8, 另一时间窗重测得 8/8 ——
    pass^k 只在本次测量窗口内有效, 跨窗口比较必须重测基线, 故每份报告带 measured_at 戳。
    拒答/判定器异常按失败计。返回逐题 {"id","n","successes","pass_at_1","pass_hat_k",
    "runs","responses","measured_at"}。"""
    report = []
    for t in tasks:
        runs, resps = [], []
        for _ in range(n):
            resp = model(prompt_prefix + t["instruction"])
            try:
                ok = bool(t["check"](str(resp))) if resp is not None else False
            except Exception:
                ok = False
            runs.append(ok)
            resps.append(None if resp is None else str(resp)[:_TRAIL_CAP])
        s = sum(runs)
        report.append({"id": t["id"], "n": n, "successes": s,
                       "pass_at_1": s / n,
                       "pass_hat_k": ce.pass_hat_k(s, n, min(k, n)),
                       "runs": runs, "responses": resps,
                       "measured_at": time.time()})
    return report


def run_paired_repeated(model_a, model_b, tasks=ALL_TASKS, n=8, k=3,
                        prompt_prefix="严格按要求输出,不要任何多余内容。要求: "):
    """交错重复配对: 每题每轮 A/B 紧邻调用,且逐轮交替谁先(ABBA)。
    紧邻 -> 漂移同时作用于两侧,配对差值仍然有效; 交替 -> 顺序/预热效应不偏向任一侧。
    对比: 先把A全测完再测B(常见做法)会把跨窗口漂移误读成系统差异 ——
    本仓库实测同一模型同一题两个时间窗得 0/8 与 8/8, 顺序测法下这会变成一个假结论。
    任一侧该轮拒答则丢弃该轮(两侧同弃,保持配对); 全轮被弃则丢弃该题。
    返回 {"rows": [...逐题双侧 successes/pass@1/pass^k, per_rep, orders, refusals...],
          "dropped": [id], "compare": 逐题 pass@1 的配对检验}。
    refusals 记录该题各侧拒答次数 —— 一侧全拒与双侧偶发不是一回事。"""
    rows, dropped = [], []
    total_refusals = {"a": 0, "b": 0}   # 循环内累计: 整题被弃的拒答也必须计入
    for t in tasks:
        per_rep, dropped_reps, orders, refusals = [], 0, [], {}
        for rep in range(n):
            prompt = prompt_prefix + t["instruction"]
            a_first = rep % 2 == 0
            if a_first:
                ra = model_a(prompt)
                rb = model_b(prompt)
            else:
                rb = model_b(prompt)
                ra = model_a(prompt)
            if ra is None or rb is None:
                dropped_reps += 1
                for s, r in (("a", ra), ("b", rb)):
                    if r is None:
                        refusals[s] = refusals.get(s, 0) + 1
                        total_refusals[s] += 1
                continue

            def ok(resp):
                try:
                    return bool(t["check"](str(resp)))
                except Exception:
                    return False
            per_rep.append((ok(ra), ok(rb)))
            orders.append(a_first)
        if not per_rep:
            dropped.append(t["id"])
            continue
        eff = len(per_rep)
        sa = sum(x for x, _ in per_rep)
        sb = sum(y for _, y in per_rep)
        rows.append({"id": t["id"], "n": eff, "dropped_reps": dropped_reps,
                     "refusals": refusals or None,   # 哪一侧拒答: 系统属性, 不该被合并计数
                     "a_successes": sa, "b_successes": sb,
                     "a_pass_at_1": sa / eff, "b_pass_at_1": sb / eff,
                     "a_pass_hat_k": ce.pass_hat_k(sa, eff, min(k, eff)),
                     "b_pass_hat_k": ce.pass_hat_k(sb, eff, min(k, eff)),
                     "per_rep": per_rep, "orders": orders,
                     "measured_at": time.time()})
    if not rows:
        raise ValueError("全部任务被丢弃,无可比数据")
    compare = ce.paired_compare([r["a_pass_at_1"] for r in rows],
                                [r["b_pass_at_1"] for r in rows])
    a_only = sum(a and not b for r in rows for a, b in r["per_rep"])
    b_only = sum(b and not a for r in rows for a, b in r["per_rep"])
    attempts = len(tasks) * n
    return {"rows": rows, "dropped": dropped, "compare": compare,
            "discordant": {"a_only": a_only, "b_only": b_only,
                           "mcnemar_p": ce.mcnemar_exact(a_only, b_only)},
            # 批次级拒答: 一侧拒答15%时, 它在剩下85%上的可比性本身需要标注
            "refusals": total_refusals, "attempts_per_side": attempts,
            "refusal_rate": {s: c / attempts if attempts else None
                             for s, c in total_refusals.items()}}


def run_interleaved(models, tasks=ALL_TASKS, n=8, k=3,
                    prompt_prefix="严格按要求输出,不要任何多余内容。要求: "):
    """N 系统交错重复: 每题每轮把所有系统紧邻调用一次, 调用顺序逐轮轮转。
    这是构建 reliability_matrix 的正确路径。反例(本仓库真实教训): 逐系统分别调
    run_repeated 等于"先把A测完再测B", 跨窗漂移与顺序效应会混进系统差异 ——
    据此曾得出一个假的"非单调能力洞"结论, 交错重测后差异归零。
    任一系统该轮拒答 -> 丢弃该轮(全系统同弃, 保持对齐); 全轮被弃 -> 丢弃该题。
    返回 {"reports": {name: run_repeated 同构报告(附 orders/refusals)}, "dropped": [id],
    "refusals"/"refusal_rate"/"attempts_per_system": 批次级拒答归属 ——
    某系统拒答15%时, 它在剩下85%上的可比性本身需要标注}。
    reports 可直接传给 reliability_matrix。"""
    names = list(models)
    if len(names) < 2:
        raise ValueError("至少需要两个系统")
    reports = {nm: [] for nm in names}
    dropped = []
    total_refusals = {nm: 0 for nm in names}   # 批次级: 整题被弃的拒答也计入
    for t in tasks:
        prompt = prompt_prefix + t["instruction"]
        oks = {nm: [] for nm in names}
        resps = {nm: [] for nm in names}
        orders, dropped_reps, refusals = [], 0, {}
        for rep in range(n):
            shift = rep % len(names)
            order = names[shift:] + names[:shift]
            got = {nm: models[nm](prompt) for nm in order}
            if any(got[nm] is None for nm in names):
                dropped_reps += 1
                for nm in names:
                    if got[nm] is None:
                        refusals[nm] = refusals.get(nm, 0) + 1
                        total_refusals[nm] += 1
                continue
            orders.append(order)
            for nm in names:
                try:
                    oks[nm].append(bool(t["check"](str(got[nm]))))
                except Exception:
                    oks[nm].append(False)
                resps[nm].append(str(got[nm])[:_TRAIL_CAP])
        if not orders:
            dropped.append(t["id"])
            continue
        eff = len(orders)
        stamp = time.time()
        for nm in names:
            s = sum(oks[nm])
            reports[nm].append({"id": t["id"], "n": eff, "successes": s,
                                "pass_at_1": s / eff,
                                "pass_hat_k": ce.pass_hat_k(s, eff, min(k, eff)),
                                "runs": oks[nm], "responses": resps[nm],
                                "dropped_reps": dropped_reps,
                                "refusals": refusals or None, "orders": orders,
                                "measured_at": stamp})
    if not any(reports[nm] for nm in names):
        raise ValueError("全部任务被丢弃,无可比数据")
    attempts = len(tasks) * n
    return {"reports": reports, "dropped": dropped,
            "refusals": total_refusals, "attempts_per_system": attempts,
            "refusal_rate": {nm: c / attempts if attempts else None
                             for nm, c in total_refusals.items()}}


def pairwise_compare(reports, max_span_s=3600, require_interleaved=True):
    """N 系统两两比较, 含多重比较校正。reports 应来自 run_interleaved。
    k 个系统产生 k(k-1)/2 次检验 —— 不校正就是"比了十对庆祝那一对显著的"。
    每对给两个读数: 逐题均值的置换检验, 逐轮不一致对的 McNemar 精确检验;
    两个家族各自独立做 Holm 校正(p_perm_holm / p_mcnemar_holm)。
    守卫与 reliability_matrix 同源(任务集一致/测量窗口/交错性)。
    返回按 |mean_diff| 降序的 [{a,b,mean_diff,p_perm,p_perm_holm,a_only,b_only,
    mcnemar_p,p_mcnemar_holm,by_task,concentration}]。by_task/concentration 揭示
    不一致对的分布: 同样的总数, 集中在一题(可能是题目问题)与散布多题(系统性差距)
    诊断完全不同, 而 p 值对两者一视同仁。"""
    reliability_matrix(reports, max_span_s=max_span_s,
                       require_interleaved=require_interleaved)  # 复用三道守卫
    names = sorted(reports)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    if not pairs:
        raise ValueError("至少需要两个系统")
    out = []
    for a, b in pairs:
        va = [r["pass_at_1"] for r in reports[a]]
        vb = [r["pass_at_1"] for r in reports[b]]
        cmp_ = ce.paired_compare(va, vb)
        a_only = b_only = 0
        by_task = []
        for ra, rb in zip(reports[a], reports[b]):
            ta = tb = 0
            for xa, xb in zip(ra.get("runs", []), rb.get("runs", [])):
                ta += bool(xa) and not bool(xb)
                tb += bool(xb) and not bool(xa)
            a_only += ta
            b_only += tb
            if ta or tb:
                by_task.append({"id": ra["id"], "a_only": ta, "b_only": tb})
        total_disc = a_only + b_only
        # 集中度: 单题贡献了多大比例的不一致对。接近1 = 单题异常(可能是题目问题),
        # 分散 = 系统性差距。总不一致对数相同但分布不同, 诊断完全不同, 而 p 值看不出来。
        concentration = (max(r["a_only"] + r["b_only"] for r in by_task) / total_disc
                         if by_task and total_disc else None)
        out.append({"a": a, "b": b, "mean_diff": cmp_["mean_diff"],
                    "p_perm": cmp_["p_value"], "diff_ci": cmp_["diff_ci"],
                    "a_only": a_only, "b_only": b_only,
                    "by_task": by_task, "concentration": concentration,
                    "mcnemar_p": ce.mcnemar_exact(a_only, b_only)})
    for key, adj in (("p_perm", "p_perm_holm"), ("mcnemar_p", "p_mcnemar_holm")):
        for row, p in zip(out, ce.holm_adjust([r[key] for r in out])):
            row[adj] = p
    return sorted(out, key=lambda r: -abs(r["mean_diff"]))


def reliability_matrix(reports, divergence=0.25, max_span_s=3600, require_interleaved=True):
    """多系统可靠性透视: 每题一行,各系统 pass@1 并排,
    spread = 跨系统最大差距, spread >= divergence 标为分歧项(真实能力差 或 题目歧义,人工分诊)。
    reports 应来自 run_interleaved(正确路径); 各系统任务集必须一致。三道守卫:
    - 任务集一致: id 序列不同直接报错。
    - 测量窗口: measured_at 齐备时校验跨度 <= max_span_s(默认1h), 可靠性跨会话漂移。
    - 交错性: 同题各系统时戳必须一致(交错测量的标志)。逐系统分别调 run_repeated 再并排
      是错误路径 —— 跨窗漂移与顺序效应会混进系统差异, 本仓库据此得出过假的"非单调洞"。
    require_interleaved=False / max_span_s=None 可显式关闭对应守卫。"""
    if not reports:
        raise ValueError("reports 不能为空")
    stamps = [r["measured_at"] for rep in reports.values() for r in rep if "measured_at" in r]
    total = sum(len(rep) for rep in reports.values())
    if max_span_s is not None and len(stamps) == total and stamps:
        span = max(stamps) - min(stamps)
        if span > max_span_s:
            raise ValueError(f"报告测量窗口跨度 {span:.0f}s 超过 {max_span_s}s: "
                             "可靠性跨会话漂移,请重测基线或显式设 max_span_s=None")
    names = sorted(reports)
    ids = [r["id"] for r in reports[names[0]]]
    for nm in names[1:]:
        if [r["id"] for r in reports[nm]] != ids:
            raise ValueError("各系统的任务集必须一致(id 序列不同)")
    if require_interleaved and len(names) > 1:
        for i in range(len(ids)):
            st = [reports[nm][i].get("measured_at") for nm in names]
            if all(s is not None for s in st) and len(set(st)) > 1:
                raise ValueError(
                    f"任务 {ids[i]} 各系统时戳不一致,判定为顺序测量(非交错): "
                    "请用 run_interleaved 交错重测,或显式 require_interleaved=False 承担混淆风险")
    rows = []
    for i, tid in enumerate(ids):
        row = {"id": tid}
        for nm in names:
            row[nm] = reports[nm][i]["pass_at_1"]
        vals = [row[nm] for nm in names]
        row["spread"] = max(vals) - min(vals)
        row["divergent"] = row["spread"] >= divergence
        rows.append(row)
    return rows


def saturation(reports, max_span_s=3600, require_interleaved=True):
    """诊断任务集的有效信息量。全系统全轮次同结果的题(恒过 / 恒败)不携带区分信息:
    McNemar 的有效样本是不一致对, 这类题的贡献恒为 0 —— 加多少道都不会提高功效。
    这比"任务数不够"更根本: MDE 应按 informative 数而非总题数来读。
    恒过 = 太容易(或判定器漏勺), 恒败 = 太难(或题目/判定器有毛病), 两者都该换题。
    返回 {"n_tasks","saturated_pass","saturated_fail","informative","informative_rate","ids"}。"""
    reliability_matrix(reports, max_span_s=max_span_s,
                       require_interleaved=require_interleaved)  # 复用三道守卫
    names = sorted(reports)
    ids = [r["id"] for r in reports[names[0]]]
    sat_pass, sat_fail, info = [], [], []
    for i, tid in enumerate(ids):
        runs = [bool(x) for nm in names for x in reports[nm][i].get("runs", [])]
        if runs and all(runs):
            sat_pass.append(tid)
        elif runs and not any(runs):
            sat_fail.append(tid)
        else:
            info.append(tid)
    n = len(ids)
    return {"n_tasks": n, "saturated_pass": len(sat_pass), "saturated_fail": len(sat_fail),
            "informative": len(info), "informative_rate": len(info) / n if n else None,
            "ids": {"saturated_pass": sat_pass, "saturated_fail": sat_fail,
                    "informative": info}}


def screen_tasks(candidates, models, n=2, confirm_n=6, **kw):
    """新题入库前的两阶段信息量筛选。
    阶段1(便宜): 全部候选交错跑 n 轮, 挑出非饱和的。
    阶段2(复核): 被挑出的再跑 confirm_n 轮, 仍非饱和才算过关 —— 低 n 筛选会把噪声
    标成"有区分力", 实测: 10 道多约束候选在 n=2 下挑出 1 道(0.5 vs 1.0),
    n=6 复核后两侧 6/6、零不一致对, 该信号纯属噪声。confirm_n=None 可跳过复核。
    另一条实测教训: 12 道"更硬"的单步字符串/算术题(10字符倒序、1234×5678、第20个素数)
    对 smol+default 全部恒过 —— 提升区分力需要难度质变, 不是同类题加长。
    candidates 需满足 canonical 通过自身 check 且 id 不与 ALL_TASKS 冲突。
    返回 {"kept","flagged","saturated_pass","saturated_fail","dropped_on_confirm",
    "reports","saturation","confirm_saturation"}。"""
    existing = {t["id"] for t in ALL_TASKS}
    for c in candidates:
        if not c.get("id") or not c.get("instruction") or "check" not in c or "canonical" not in c:
            raise ValueError(f"候选题字段不全: {c.get('id')!r}")
        if c["id"] in existing:
            raise ValueError(f"候选题 id 与现有任务冲突: {c['id']}")
        if not c["check"](c["canonical"]):
            raise ValueError(f"候选题 canonical 过不了自身判定器: {c['id']}")
    run = run_interleaved(models, tasks=candidates, n=n, **kw)
    sat = saturation(run["reports"])
    by_id = {c["id"]: c for c in candidates}
    flagged = [by_id[i] for i in sat["ids"]["informative"]]
    out = {"flagged": flagged,
           "saturated_pass": [by_id[i] for i in sat["ids"]["saturated_pass"]],
           "saturated_fail": [by_id[i] for i in sat["ids"]["saturated_fail"]],
           "reports": run["reports"], "saturation": sat,
           "confirm_saturation": None, "dropped_on_confirm": []}
    if confirm_n is None or not flagged:
        out["kept"] = list(flagged)
        return out
    run2 = run_interleaved(models, tasks=flagged, n=confirm_n, **kw)
    sat2 = saturation(run2["reports"])
    kept_ids = set(sat2["ids"]["informative"])
    out["kept"] = [t for t in flagged if t["id"] in kept_ids]
    out["dropped_on_confirm"] = [t for t in flagged if t["id"] not in kept_ids]
    out["confirm_saturation"] = sat2
    return out


# ---------------------------------------------------------------- 轨迹门禁 fixture

# claim_eval.trajectory_selfcheck 的开箱可用样例(本仓库实测产出, 已验证自洽:
# 忠实回答的 8 条 claim 全部 grounded(零误报), 两个篡改变体各自命中期望类别)。
# 没有 fixture 的门禁等于不会被跑 —— distorted 类在真实数据上极少自然出现,
# 必须靠人工篡改测灵敏度, 否则"零歪曲"分不清是模型忠实还是检测器瞎了。
_GATE_OBS = [{"tool_call_id": "tc_1", "tool": "web_search", "observation": "[1] 字节跳动2023年营收被曝达1100亿美元,赶超腾讯逼近Meta (36kr.com)\n    12月20日,据外媒报道,字节跳动2023年销售额达到1100亿美元,超过腾讯。对此,字节跳动方面并未做出回应。实际上,字节跳动今年营收超过腾讯已是预料之中的事。11月,据The Information报道,字节跳动二季度收入增长超过40%,达290亿美元;上半年营收约为540亿美元。去年,字节跳动年收入为850亿美元。"}]
_GATE_FAITHFUL = "据外媒2023年12月20日报道，字节跳动2023年销售额达到1100亿美元，超过腾讯，逼近Meta；对此字节跳动方面未作回应。此前11月The Information报道称，其二季度收入同比增长超过40%，达290亿美元，上半年营收约540亿美元。相比2022年全年850亿美元的收入，其营收规模明显扩大，因此超越腾讯被视为预料之中的结果。以上数据均来自外媒报道，未经字节跳动官方确认。"
_GATE_DISTORTED = "据外媒2023年12月20日报道，字节跳动2023年销售额达到2000亿美元，超过腾讯，逼近Meta；对此字节跳动方面未作回应。此前11月The Information报道称，其二季度收入同比增长超过200%，达290亿美元，上半年营收约900亿美元。相比2022年全年850亿美元的收入，其营收规模明显扩大，因此超越腾讯被视为预料之中的结果。以上数据均来自外媒报道，未经字节跳动官方确认。"
_GATE_FABRICATED = "据外媒2023年12月20日报道，字节跳动2023年销售额达到1100亿美元，超过腾讯，逼近Meta；对此字节跳动方面未作回应。此前11月The Information报道称，其二季度收入同比增长超过40%，达290亿美元，上半年营收约540亿美元。相比2022年全年850亿美元的收入，其营收规模明显扩大，因此超越腾讯被视为预料之中的结果。以上数据均来自外媒报道，未经字节跳动官方确认。此外,字节跳动2023年净利润为500亿美元,员工总数达20万人。"

TRAJECTORY_GATE = [{
    "observations": _GATE_OBS,
    "faithful": _GATE_FAITHFUL,
    "planted": [
        {"response": _GATE_DISTORTED, "expect": "distorted", "desc": "三处数字被放大(1100->2000亿, 40%->200%, 540->900亿)"},
        {"response": _GATE_FABRICATED, "expect": "fabricated", "desc": "追加资料中不存在的净利润与员工数"},
    ],
}]


# ---------------------------------------------------------------- world / rubric 门禁 fixture

# claim_eval.selfcheck 的开箱样例(本会话实测两次, recall 2/2, 干净文档 fp 0)。
# 注意与 TRAJECTORY_GATE 的本质差别: world 门禁依赖实时检索, 是"会腐坏"的 fixture ——
# 事实会变(高管变动、营收更新)、检索结果会漂移, 因此需定期复验并更新 planted 真值;
# trajectory/rubric fixture 自洽封闭(证据就在 case 里), 不会因外部世界变化而失效。
WORLD_GATE = [
    {"text": "腾讯公司成立于1998年,创始人包括马化腾和张志东。微信于2013年1月发布。"
             "腾讯总部位于深圳。2023年腾讯全年营收约为8500亿元人民币。",
     "planted": [{"substr": "2013", "desc": "微信发布年份(真值2011年1月)"},
                 {"substr": "8500", "desc": "2023营收8500亿(真值约6090亿)"}]},
    # planted 为空 = 干净文档, 用于测误报率(缺了它只能测查全, 无法发现"来者皆错"的判定器)
    {"text": "腾讯公司成立于1998年11月,总部位于深圳。马化腾是腾讯的创始人之一。"
             "微信是腾讯旗下的即时通讯产品。", "planted": []},
]

# rubric_eval.rubric_canary 的开箱样例(本会话实测: good=1.00 fooling=0.00 separation=1.00)。
# fooling 是"空洞奉承"型回答: 语气专业、零具体信息 —— 能被它骗到的 rubric 不该上线。
RUBRIC_GATE = {
    "criteria": [
        {"text": "回答给出了2023年腾讯研发投入的具体金额", "weight": 3},
        {"text": "回答说明了研发投入的同比变化幅度", "weight": 2},
        {"text": "回答为关键数据标注了信息来源", "weight": 1},
    ],
    "good": "根据腾讯2023年报,2023年研发投入为640亿元,同比增长约4%。",
    "fooling": "本回答对腾讯的研发投入情况进行了全面深入且详实的分析,视角专业,"
               "结论可靠,极具参考价值。",
}


# ---------------------------------------------------------------- harness 评测子集

# 经实测确认对脚手架敏感的题(供固定模型比脚手架时选用)。
# 依据: default 固定, 严格脚手架 vs 裸指令, 交错重复 n=10 与 n=6 两轮 ——
#   if-json  strict 6/6 vs bare 2/6, pass^3 1.00 vs 0.00, 两轮一致(裸指令输出 markdown 围栏)
#   if-3lines 首轮 10/10 vs 8/10, 复核 6/6 vs 6/6 -> 弱信号, 故不列入
# 其余 IF 题两侧恒过(saturation 报 2/8 有信息), 放进 harness A/B 只是烧钱。
# 新增条目前请照同样流程实测+复核: 猜测哪些题"应该"敏感, 实测会打脸(本例 7/8 猜错)。
SCAFFOLD_SENSITIVE = ["if-json", "fmt-jsonarr", "fmt-kv", "fmt-yaml"]

# 扩充依据(第二轮筛选, 8 道格式脆弱候选, 严格脚手架 vs 裸指令, 两阶段 n=2 筛 + n=6 复核):
#   保留 3 道: fmt-jsonarr(2/2 vs 0/2) fmt-kv(2/2 vs 1/2) fmt-yaml(2/2 vs 0/2)
#   拒 5 道: fmt-csv/oneword/nounit/nopunct/bare-num 两侧恒过 —— 单值输出不易被污染
# 共同失败形态: ```json ... ``` 围栏, 或 "形式如下:" 之类前置解释。
# 规律: 脚手架敏感度来自"多行/结构化输出", 单值答案几乎不敏感 —— 又一次猜测被实测收窄
# (原以为 8 道格式题都敏感, 实际 3/8)。


def report(reports, alpha=0.05, divergence=0.25, refusals=None, **kw):
    """把一次交错测量整理成完整诚实报告 —— 每一项都曾在手工报告里被漏掉过。
    reports: run_interleaved 的 reports(也接受任何同构的 {name: 逐题报告})。
    每对给出: 效应量+CI、逐题置换检验、逐轮 McNemar、Holm 校正、不一致对分布与
    集中度(单题异常 vs 跨题复现)、以及 null 分支的可排除范围(claim_eval.interpret)。
    另给全局: 有效样本(饱和诊断)、拒答归属、以及触顶/触底系统的清单 —— 后者是第114轮
    踩到的坑: 2x2 里三个格子都是 1.000 时, 交互项 -0.667 无法与天花板假象区分。
    返回 {"pairs","saturation","ceiling","text"}; ceiling={"at_top":[...],"at_bottom":[...]}。"""
    sat = saturation(reports, **kw)
    rows = pairwise_compare(reports, **kw)
    rates = {nm: (sum(x["successes"] for x in v) / n if (n := sum(x["n"] for x in v)) else None)
             for nm, v in reports.items()}
    ceiling = {"at_top": sorted(nm for nm, r in rates.items() if r == 1.0),
               "at_bottom": sorted(nm for nm, r in rates.items() if r == 0.0)}
    lines = [f"有效样本: {sat['informative']}/{sat['n_tasks']} 题有信息"
             f"(恒过 {sat['saturated_pass']}, 恒败 {sat['saturated_fail']})"]
    if refusals:
        lines.append(f"拒答: {refusals}")
    for edge, where in (("at_top", "触顶(1.000)"), ("at_bottom", "触底(0.000)")):
        names = ceiling[edge]
        if not names:
            continue
        warn = (" —— 这些系统之间的 null 与任何交互项都不可解释: 无余量可加/可减"
                if len(names) > 1 else " —— 该系统无余量, 它作为参照时效应会被压缩")
        lines.append(f"{where}: {', '.join(names)}{warn}")
    pairs = []
    for r in rows:
        # 逐轮 McNemar 是主检验(保留重复结构), 故 null 界按逐轮单元数算。
        # 合成报告若无 runs 字段则单元数为 0 -> interpret 会如实判为"无信息的 null",
        # 这比猜一个数更诚实(初版写了 `or r.get("n")` 回退, 但 compare 行里没有 n 键, 是死代码)。
        units = sum(len(x.get("runs", [])) for x in reports[r["a"]])
        fake = {"n": units, "mean_diff": r["mean_diff"], "diff_ci": r["diff_ci"],
                "p_value": r["p_mcnemar_holm"]}
        verdict = ce.interpret(fake, n_units=units, alpha=alpha)
        pairs.append({**r, "units": units, "interpretation": verdict})
        conc = "n/a" if r["concentration"] is None else f"{r['concentration']:.2f}"
        lines.append(
            f"{r['a']} vs {r['b']}: Δ={r['mean_diff']:+.3f} "
            f"CI95=[{r['diff_ci'][0]:+.3f},{r['diff_ci'][1]:+.3f}] | "
            f"逐题p={r['p_perm']:.3f} 逐轮McNemar={r['mcnemar_p']:.5f} Holm={r['p_mcnemar_holm']:.5f} | "
            f"不一致对 {r['a_only']}:{r['b_only']} 集中度={conc} | {verdict['text']}")
    return {"pairs": pairs, "saturation": sat, "ceiling": ceiling,
            "text": "\n".join(lines)}
