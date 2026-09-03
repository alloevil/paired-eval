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
    return {"rows": rows, "dropped": dropped, "dropped_detail": dropped_detail,
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
