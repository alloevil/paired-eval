# -*- coding: utf-8 -*-
"""paired_bench — 本仓库自带的配对 A/B 冒烟任务集与工作流(实战沉淀,全程序判定)。

任务 schema: {"id", "instruction", "check": str->bool, "canonical": 一个合法输出样例}
model 依赖注入: model(prompt: str) -> str | None (None = 拒答/不可用 -> 该题成对丢弃)。

run_paired(model_a, model_b, tasks) 执行完整配对工作流:
逐题双侧作答 -> 程序判定 -> 成对丢弃缺失 -> claim_eval.paired_compare 出统计。
判定器故意严格(strip 后精确比较): 测的就是指令遵循,宽松即失真。
"""

import answer_match as am
import claim_eval as ce


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
    {"id": "if-nomoon", "instruction": "用一句话介绍月亮,但全句不得出现'月'这个字",
     "check": lambda r: "月" not in r and len(r.strip()) >= 5,
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


def run_paired(model_a, model_b, tasks=ALL_TASKS, prompt_prefix="严格按要求输出,不要任何多余内容。要求: "):
    """完整配对 A/B: 双侧作答 -> 程序判定 -> 任一侧 None 成对丢弃 -> 置换检验。
    返回 {"rows": [{"id", "a", "b"}], "dropped": [id], "compare": paired_compare结果}。"""
    rows, dropped = [], []
    for t in tasks:
        ra, rb = model_a(prompt_prefix + t["instruction"]), model_b(prompt_prefix + t["instruction"])
        if ra is None or rb is None:
            dropped.append(t["id"])
            continue

        def score(resp):
            try:
                return 1.0 if t["check"](str(resp)) else 0.0
            except Exception:
                return 0.0   # 输出连判定器都解析不了 = 不合规
        rows.append({"id": t["id"], "a": score(ra), "b": score(rb)})
    if not rows:
        raise ValueError("全部任务被成对丢弃,无可比数据")
    compare = ce.paired_compare([r["a"] for r in rows], [r["b"] for r in rows])
    return {"rows": rows, "dropped": dropped, "compare": compare}
