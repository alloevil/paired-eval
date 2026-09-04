# -*- coding: utf-8 -*-
"""L1 答案匹配器 — "回答中带有真值"的类型化验证。

配套 claim_eval.py 的另一半:有真值任务的 约束提取 -> 类型化匹配 -> 三值判分。
零依赖、纯函数、确定性。exact string match 几乎总是错的 —— 按真值类型选匹配器:
    parse_number("8635亿")            # 中文单位/百分比/分数/千分位 -> float
    match_numeric(pred, gold)         # 容差匹配(相对+绝对)
    match_choice(pred, golds)         # 归一化(NFKC/大小写/标点) + 别名表
    match_set(preds, golds)           # 集合等价 + F1 部分分
    extract_answer(text)              # 槽位提取: 答案:X / \\boxed{X}, 失败=格式错
    grade_answer(response, gold, ...) # SimpleQA 式三值: correct/incorrect/not_attempted
弃答(not_attempted)必须单列 —— 只报二分类会奖励瞎猜。
"""

import re
import unicodedata

# ---------------------------------------------------------------- 归一化

_PUNCT_RE = re.compile(r"[\s。，,.;；、：:！!？?\"'()（）\[\]【】]+")


def normalize_text(s):
    """NFKC(全角->半角) + casefold + 去空白与常见标点。用于选项/实体比对,不用于数值。"""
    return _PUNCT_RE.sub("", unicodedata.normalize("NFKC", str(s)).casefold())


# ---------------------------------------------------------------- 数值

_FRACTION_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_UNITS = (("万亿", 1e12), ("亿", 1e8), ("万", 1e4))


def parse_number(s):
    """从短文本解析数值:千分位/中文单位(万/亿/万亿)/百分号/分数。取第一个数字,
    单位取其后紧邻字符。解析失败返回 None。面向答案槽位等短字符串,不是通用NLP。"""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    frac = _FRACTION_RE.match(s)
    if frac:
        return float(frac.group(1)) / float(frac.group(2))
    m = _NUM_RE.search(s)
    if not m:
        return None
    value = float(m.group(0).replace(",", ""))
    tail = s[m.end():m.end() + 3]
    if tail.startswith("%") or tail.startswith("％"):
        return value / 100.0
    for unit, mult in _UNITS:
        if tail.startswith(unit):
            return value * mult
    return value


def match_numeric(pred, gold, rel_tol=0.01, abs_tol=0.0):
    """数值容差匹配。两边都先 parse_number;任一解析失败 -> False。"""
    a, b = parse_number(pred), parse_number(gold)
    if a is None or b is None:
        return False
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# ---------------------------------------------------------------- 选项/实体/集合

def match_choice(pred, golds):
    """归一化后与任一可接受答案相等。golds: 单个真值或别名列表(如 ["北京","Beijing"])。"""
    accept = [golds] if isinstance(golds, str) else list(golds)
    p = normalize_text(pred)
    return any(p == normalize_text(g) for g in accept)


def match_set(preds, golds, item_match=None):
    """集合比对(顺序无关):exact + F1 部分分。item_match 默认归一化相等。"""
    eq = item_match or (lambda a, b: normalize_text(a) == normalize_text(b))
    matched_gold = set()
    hit = 0
    for p in preds:
        for j, g in enumerate(golds):
            if j not in matched_gold and eq(p, g):
                matched_gold.add(j)
                hit += 1
                break
    precision = hit / len(preds) if preds else 0.0
    recall = hit / len(golds) if golds else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"exact": hit == len(preds) == len(golds), "precision": precision,
            "recall": recall, "f1": f1}


# ---------------------------------------------------------------- 提取与判分

def _last_boxed(text):
    """取最后一个 \\boxed{...} 的内容(花括号配平,支持嵌套如 \\frac{1}{2})。
    返回 (内容, 起始位置) 或 (None, -1)。正则 [^{}]* 会在嵌套时静默失配 —— 手工扫描。"""
    head = "\\boxed{"
    i = text.rfind(head)
    if i < 0:
        return None, -1
    depth, j = 0, i + len(head)
    start = j
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[start:j], i
            depth -= 1
        j += 1
    return None, -1  # 未闭合视为无 boxed


def extract_answer(text, marker="答案"):
    """槽位提取:取最后一个 `marker: X`(至行尾) 或 \\boxed{X}。都没有返回 None。
    None 记作格式错,与答错分开统计 —— 否则你在测指令遵循而不是能力。"""
    cands = []
    content, pos = _last_boxed(text)
    if content is not None:
        cands.append((pos, content.strip()))
    slot = re.findall(rf"{re.escape(marker)}\s*[:：]\s*(.+)", text)
    if slot:
        cands.append((text.rfind(slot[-1]), slot[-1].strip()))
    if not cands:
        return None
    return max(cands)[1]  # 取出现位置最靠后的


def grade_answer(response, gold, kind="text", marker="答案", rel_tol=0.01):
    """SimpleQA 式三值判分。kind: "numeric" | "text"(选项/实体,gold 可为别名列表)。
    集合类真值请直接用 match_set(需要结构化的预测列表,自由文本切分不可靠)。
    not_attempted 带 reason 细分(只报机械可观测的状态,不猜意图):
      "blank"   = 回答本身为空/纯空白
      "no_slot" = 有实质内容但没有可解析的答案槽位
    后者混进弃答会让"格式不合规"与"拒答"不可区分, 而它们是两种不同诊断:
    no_slot 高 = 在测指令遵循而非能力(该收紧提示或放宽提取), blank 高 = 模型真的没答。"""
    extracted = extract_answer(response, marker)
    if extracted is None or not extracted.strip():
        reason = "blank" if not str(response).strip() else "no_slot"
        return {"verdict": "not_attempted", "extracted": None, "reason": reason}
    if kind == "numeric":
        ok = match_numeric(extracted, gold, rel_tol=rel_tol)
    else:
        ok = match_choice(extracted, gold)
    return {"verdict": "correct" if ok else "incorrect", "extracted": extracted, "reason": None}


def aggregate_grades(grades):
    """三值聚合:accuracy 的分母是 attempted,弃答率单列并按 reason 细分。"""
    n = len(grades)
    cor = sum(g["verdict"] == "correct" for g in grades)
    att = sum(g["verdict"] != "not_attempted" for g in grades)
    blank = sum(g.get("reason") == "blank" for g in grades)
    no_slot = sum(g.get("reason") == "no_slot" for g in grades)
    return {"n": n, "correct": cor, "attempted": att,
            "accuracy_on_attempted": cor / att if att else None,
            "overall_accuracy": cor / n if n else None,
            "not_attempted_rate": (n - att) / n if n else None,
            "blank": blank, "no_slot": no_slot,
            "no_slot_rate": no_slot / n if n else None}
