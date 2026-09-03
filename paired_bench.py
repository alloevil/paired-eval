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
判定器故意严格(strip 后精确比较): 测的就是指令遵循,宽松即失真。
纪律: 单发分歧项必须经 run_repeated 复核才许下结论(有误标前科);
多系统比较必须走交错路径, 逐系统分别 run_repeated 再并排会被 reliability_matrix 拒绝。
功效上限(诚实披露): ALL_TASKS 只有 28 题, claim_eval.detectable_effect(28)≈0.27 ——
即"较优系统需在 ≥27% 的题上单方面胜出且几乎无反向失误"才可能显著。这是冒烟集,
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

ALL_TASKS = EXACT_QA + IF_TASKS + CHAR_TASKS


_TRAIL_CAP = 120  # 留痕截断长度: 够诊断失败模式,不撑爆报告


def run_paired(model_a, model_b, tasks=ALL_TASKS, prompt_prefix="严格按要求输出,不要任何多余内容。要求: "):
    """完整配对 A/B: 双侧作答 -> 程序判定 -> 任一侧 None 成对丢弃 -> 置换检验。
    调用顺序逐题交替(偶数题先A,奇数题先B): 固定先A后B会把窗口内漂移与预热效应
    系统性地压在同一侧 —— 与 judge 评测里交换位置消除 position bias 同理。
    返回 {"rows": [{"id","a","b","resp_a","resp_b","a_first"}], "dropped": [id], "compare": ...}。
    原始输出留痕(截断): 分歧项诊断靠看失败输出长什么样,只留 0/1 无法审计。"""
    rows, dropped = [], []
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
    return {"rows": rows, "dropped": dropped, "compare": compare}


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
    返回 {"rows": [...逐题双侧 successes/pass@1/pass^k, per_rep, orders...],
          "dropped": [id], "compare": 逐题 pass@1 的配对检验}。"""
    rows, dropped = [], []
    for t in tasks:
        per_rep, dropped_reps, orders = [], 0, []
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
    return {"rows": rows, "dropped": dropped, "compare": compare,
            "discordant": {"a_only": a_only, "b_only": b_only,
                           "mcnemar_p": ce.mcnemar_exact(a_only, b_only)}}


def run_interleaved(models, tasks=ALL_TASKS, n=8, k=3,
                    prompt_prefix="严格按要求输出,不要任何多余内容。要求: "):
    """N 系统交错重复: 每题每轮把所有系统紧邻调用一次, 调用顺序逐轮轮转。
    这是构建 reliability_matrix 的正确路径。反例(本仓库真实教训): 逐系统分别调
    run_repeated 等于"先把A测完再测B", 跨窗漂移与顺序效应会混进系统差异 ——
    据此曾得出一个假的"非单调能力洞"结论, 交错重测后差异归零。
    任一系统该轮拒答 -> 丢弃该轮(全系统同弃, 保持对齐); 全轮被弃 -> 丢弃该题。
    返回 {"reports": {name: run_repeated 同构报告(附 orders)}, "dropped": [id]},
    reports 可直接传给 reliability_matrix。"""
    names = list(models)
    if len(names) < 2:
        raise ValueError("至少需要两个系统")
    reports = {nm: [] for nm in names}
    dropped = []
    for t in tasks:
        prompt = prompt_prefix + t["instruction"]
        oks = {nm: [] for nm in names}
        resps = {nm: [] for nm in names}
        orders, dropped_reps = [], 0
        for rep in range(n):
            shift = rep % len(names)
            order = names[shift:] + names[:shift]
            got = {nm: models[nm](prompt) for nm in order}
            if any(got[nm] is None for nm in names):
                dropped_reps += 1
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
                                "dropped_reps": dropped_reps, "orders": orders,
                                "measured_at": stamp})
    if not any(reports[nm] for nm in names):
        raise ValueError("全部任务被丢弃,无可比数据")
    return {"reports": reports, "dropped": dropped}


def pairwise_compare(reports, max_span_s=3600, require_interleaved=True):
    """N 系统两两比较, 含多重比较校正。reports 应来自 run_interleaved。
    k 个系统产生 k(k-1)/2 次检验 —— 不校正就是"比了十对庆祝那一对显著的"。
    每对给两个读数: 逐题均值的置换检验, 逐轮不一致对的 McNemar 精确检验;
    两个家族各自独立做 Holm 校正(p_perm_holm / p_mcnemar_holm)。
    守卫与 reliability_matrix 同源(任务集一致/测量窗口/交错性)。
    返回按 |mean_diff| 降序的 [{a,b,mean_diff,p_perm,p_perm_holm,a_only,b_only,
    mcnemar_p,p_mcnemar_holm}]。"""
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
        for ra, rb in zip(reports[a], reports[b]):
            for xa, xb in zip(ra.get("runs", []), rb.get("runs", [])):
                a_only += bool(xa) and not bool(xb)
                b_only += bool(xb) and not bool(xa)
        out.append({"a": a, "b": b, "mean_diff": cmp_["mean_diff"],
                    "p_perm": cmp_["p_value"], "diff_ci": cmp_["diff_ci"],
                    "a_only": a_only, "b_only": b_only,
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
