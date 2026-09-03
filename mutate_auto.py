#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""系统化变异测试: AST 遍历机械生成变异, 消除手挑变异的选择偏差。

手挑变异只能测到"我想到的地方"(本仓库前三批: 自选10个0存活, 故意找茬则7个存活)。
本工具枚举所有可变异节点(比较符/布尔运算/数值常量/not), 逐个应用并跑套件。

必做对照: 先验证 ast.unparse 空变异后套件仍全绿 —— 否则 unparse 自身的改写
(丢注释、改引号、规范化)会让所有变异都"被杀死", 得出虚假的满分。

用法: python3 mutate_auto.py [模块...] [--limit N] [--seed S]
     python3 mutate_auto.py --baseline   # 全量跑并把已确认的等价变异写入基线
     python3 mutate_auto.py --check      # 全量跑并与基线比对, 新增存活即失败(棘轮)
     python3 mutate_auto.py --changed    # 只变异未提交改动涉及的行(开发中用, 约20秒)
     python3 mutate_auto.py --since=HEAD~3   # 只变异指定 ref 以来改动的行


存活变异必须分类, 不该盲目追 100%:
  真缺口   —— 存在能区分原版与变异版的输入, 但套件里没有。例(本仓库实测):
              match_numeric 的 <=/< 差别只在"差值恰好等于容差"时显现;
              rubric_canary 的 (bs or 0.0) 只在糊弄回答不可判定时显现。
  等价变异 —— 语义上无法区分, 任何输入都同样表现。例(本仓库实测):
              parse_number 的单位窗口 3->4(单位最长2字符)、哨兵位置 -1->-2
              (content 为 None 时该值从不被使用)、日志截断 80->81。
等价变异应记录并解释, 而不是为它编造测试 —— 那种测试只是把实现细节钉死。

claim_eval.py 全量扫描(165点)后剩余 13 个存活, 全部经分析为等价变异, 分五类:
  截断/上限常量      EVIDENCE_CAP 6000, round(·,10) 的 10  —— 只影响长度/精度余量
  被后续条件掩盖    range(max_retries+1) 的上界被 `attempt == max_retries` 的 break 支配
  死初始化          adjusted = [0.0]*n 每个位置随后都被覆盖
  零守卫的等价分支  `n = len(results) or 1` 里 1 换成 2: 分子为 0, 商仍为 0
  实践中不可达      p 值恰好等于 alpha、浮点恰好等于 mean_diff-1e-12、
                    p_win<=2 被 `p_win+p_loss>1` 的检查支配
  蒙特卡洛计数微调  reproduce_findings 的 sims=200->201: 两者都给 required_tasks=81,
                    对输出不可观测(把 sims 钉死等于禁止将来调精度)
  被前置条件算术否决  screen_graded 复核判据里的 `s >= hi` 换 `s > hi`: 要产生差异需
                    "全部轮次恰等于 hi 且均分严格小于 hi", 而每轮都等于 hi 则均分必
                    等于 hi -- 算术上不可能, 且"均分在区间内"这一条已先否决
  被显著性掩盖      check_derivation_increment 的 `mean_diff < 0 and 显著`: mean_diff
                    恰为 0 时置换检验的 p 必为 1.0, 永不显著, 故 < 换 <= 不可观测
  上限阀门的等价边界  p_floor / min_units_for_alpha 的 n<64 与 n>64 安全阀: n>=64 时
                    2/2^n 已远低于重采样地板 1/(resamples+1), max() 恒取后者, 故 64
                    换 65、`<` 换 `<=`、`>` 换 `>=` 都不可观测。起点 n=1 换 2 同理:
                    p_floor(1)=1.0 >= 任何 alpha<1, 循环必至少走一步。

分类方法(第115轮定下, 比逐个猜快得多): 对每个存活变异, 把变异体 exec 成模块, 用一组
探针输入(p_floor 跑 n=1..100 x 两种 resamples、min_units 跑 11 个 alpha、interpret 跑
7 组 (n,alpha))算出输出向量, 与基线逐项比对。零差异 = 等价, 有差异则差异本身就指出
该断言什么。实测: 11 个存活里 5 个零差异(等价), 5 个有差异(真缺口, 补测后全部杀死),
1 个是死代码(`n >= 1` 在提前返回后恒真, 删掉即消失)。
这类结论应记录而非补测: 为它们写的测试只会把实现细节钉死, 让合理重构无谓失败。
"""
import ast
import json
import re
import pathlib
import random
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent

CMP_FLIP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
            ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}


def _noise_constants(tree):
    """已知等价类的常量节点 id: 变异它们只会产出无法区分的变异体, 纯噪声。
    - 函数默认参数值(sims/seed/max_workers/n/confirm_n/窗口秒数...): 是约定不是契约,
      测试若把它们钉死, 将来调参就会无谓失败。
    - 切片截断长度(日志/留痕的 [:80]、[:120]): 只影响消息长短。
    实测: 前两轮扫描的存活变异里, 这两类占了 10/11。"""
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in list(node.args.defaults) + [d for d in node.args.kw_defaults if d]:
                for sub in ast.walk(d):
                    skip.add(id(sub))
        elif isinstance(node, ast.Subscript):
            for sub in ast.walk(node.slice):
                skip.add(id(sub))
    return skip


def _scopes(tree):
    """行号 -> 所在函数限定名(取最内层)。用于给变异点一个编辑稳定的标识:
    行号会随任何上方插入而平移, 基线里 13 条已知等价变异会集体变成"新增存活",
    旧条目同时显示"已修补" —— 棘轮第一次被使用就会因此误报。
    函数名 + 同类出现序号只在该函数自身结构变化时才变, 正是我们想要的敏感度。"""
    spans = []
    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                spans.append((child.lineno, child.end_lineno or child.lineno, name))
                walk(child, name + ".")
            else:
                walk(child, prefix)
    walk(tree, "")
    return spans


class _Mutator(ast.NodeTransformer):
    """把第 target 个可变异点改掉; 其余原样。
    applied 形如 "wilson_ci#0: LtE->Lt": 函数限定名 + 该函数内同类变异的出现序号,
    不含行号 —— 基线标识必须能扛住无关位置的编辑。"""

    def __init__(self, target, skip_ids=frozenset(), scopes=()):
        self.target = target
        self.skip_ids = skip_ids
        self.scopes = scopes
        self.seen = 0
        self.applied = None
        self.hit_line = None      # 仅供 --changed 过滤用, 不进标识(标识必须编辑稳定)
        self._counts = {}

    def _label(self, desc, lineno):
        inner = ""
        for start, end, name in self.scopes:
            if start <= lineno <= end and len(name) >= len(inner):
                inner = name
        key = (inner or "<module>", desc)
        idx = self._counts.get(key, 0)
        self._counts[key] = idx + 1
        return f"{key[0]}#{idx}: {desc}"

    def _hit(self, desc, lineno):
        label = self._label(desc, lineno)
        if self.seen == self.target:
            self.applied = label
            self.hit_line = lineno
            self.seen += 1
            return True
        self.seen += 1
        return False

    def visit_Compare(self, node):
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            flip = CMP_FLIP.get(type(op))
            if flip and self._hit(f"{type(op).__name__}->{flip.__name__}", node.lineno):
                node.ops[i] = flip()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        swap = ast.Or if isinstance(node.op, ast.And) else ast.And
        if self._hit(f"{type(node.op).__name__}->{swap.__name__}", node.lineno):
            node.op = swap()
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self._hit("drop not", node.lineno):
            return node.operand
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return node
        if id(node) in self.skip_ids:
            return node
        if self._hit(f"const {node.value!r}->{node.value + 1!r}", node.lineno):
            return ast.Constant(value=node.value + 1)
        return node


def count_points(tree, skip_ids=frozenset()):
    m = _Mutator(-1, skip_ids, _scopes(tree))
    m.visit(tree)
    return m.seen


def suite_passes():
    return subprocess.run(["sh", str(ROOT / "runtests.sh")],
                          capture_output=True).returncode == 0


def changed_lines(mod, since="HEAD"):
    """git diff 里该文件的新增/修改行号集合。空集表示未改动。
    用于把变异范围收窄到"刚写的代码" —— 全量扫描要 6 分钟不会有人在开发中跑,
    而"你刚写的这几十行有测试吗"只需十几个变异点、约20秒, 才真正可用。"""
    out = subprocess.run(["git", "diff", "-U0", since, "--", mod],
                         cwd=ROOT, capture_output=True, text=True)
    lines = set()
    for hunk in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", out.stdout, re.M):
        start = int(hunk.group(1))
        count = int(hunk.group(2) or 1)
        lines.update(range(start, start + count))
    return lines


def run(modules, limit=None, seed=0, since=None):
    """全程 try/finally 保护: 变异会真的改写源文件, 中断(Ctrl-C/超时/异常)若不还原,
    仓库会停在被变异的状态 —— 本工具第一次全量运行就因工具层超时踩到过,
    留下一个被 unparse 的 paired_bench.py, 而后续运行的"空变异对照"如实报告了不可信。
    since 非空时只变异该 git ref 以来改动过的行(未改动的模块整个跳过)。"""
    results = []
    for mod in modules:
        path = ROOT / mod
        original = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(original)
            # 对照: 空变异(仅 unparse)必须仍然全绿
            path.write_text(ast.unparse(tree), encoding="utf-8")
            baseline_ok = suite_passes()
        finally:
            path.write_text(original, encoding="utf-8")
        if not baseline_ok:
            print(f"!! {mod}: unparse 空变异即失败, 该模块的变异结果不可信(跳过)")
            continue
        tree_c = ast.parse(original)          # skip_ids 依赖节点对象同一性: 必须同一棵树
        total = count_points(tree_c, _noise_constants(tree_c))
        idxs = list(range(total))
        scope = ""
        if since is not None:
            touched = changed_lines(mod, since)
            if not touched:
                continue                      # 该模块未改动
            keep = []
            for i in idxs:
                probe_tree = ast.parse(original)   # 与 skip 集同一棵树: 否则索引空间错位,
                probe = _Mutator(i, _noise_constants(probe_tree),      # 会变异到别的点上
                                 _scopes(probe_tree))
                probe.visit(probe_tree)
                if probe.applied and probe.hit_line in touched:
                    keep.append(i)
            idxs = keep
            scope = f", 限定 {since} 以来改动的 {len(touched)} 行"
            if not idxs:
                print(f"== {mod}: 改动行内无可变异点{scope}")
                continue
        if limit and limit < len(idxs):
            idxs = sorted(random.Random(seed).sample(idxs, limit))
        print(f"== {mod}: {total} 个可变异点(已滤除默认参数/切片长度等已知等价类), "
              f"本次跑 {len(idxs)} 个{scope} (unparse 对照通过)")
        try:
            for i in idxs:
                tree_i = ast.parse(original)
                mut = _Mutator(i, _noise_constants(tree_i), _scopes(tree_i))
                new_tree = mut.visit(tree_i)
                if mut.applied is None:
                    continue
                try:
                    src = ast.unparse(ast.fix_missing_locations(new_tree))
                except Exception as exc:
                    print(f"  skip  {mut.applied} (unparse失败: {exc})")
                    continue
                path.write_text(src, encoding="utf-8")
                killed = not suite_passes()
                results.append((mod, mut.applied, killed))
                if not killed:
                    print(f"  SURVIVED  {mod} {mut.applied}")
        finally:
            path.write_text(original, encoding="utf-8")   # 中断/异常也必须还原
    killed = sum(1 for _, _, k in results if k)
    print(f"\n杀伤率: {killed}/{len(results)}")
    for mod, desc, k in results:
        if not k:
            print(f"  存活: {mod} {desc}")
    return results


BASELINE = ROOT / "mutation_baseline.json"


def _survivors(results):
    return sorted(f"{mod}|{desc}" for mod, desc, killed in results if not killed)


def check_against_baseline(results, write=False, partial=False):
    """棘轮: 新增存活变异即失败。杀伤率是一次性成就, 除非把它钉成基线 ——
    否则新增无测试保护的代码会让存活数静默上涨, 而 pre-commit 只看"测试通过"。
    基线里记的是已分析确认的等价变异清单(见本文件顶部的五类分类)。
    两层作用域约束(都是实测踩过的坑):
    - 比对限定在本次跑过的模块: 拿部分结果与全库基线比, 未跑模块会被误报"已修补"。
    - partial(--since 只扫改动行)时: 同一模块未被扫到的已知条目同样会显示"已修补",
      按提示更新基线就会永久抹掉它们。故 partial 下只报新增、不报"已修补",
      且禁止写基线 —— 基线只能由全量扫描更新。"""
    now = _survivors(results)
    ran = sorted({mod for mod, _, _ in results})
    old = (set(json.loads(BASELINE.read_text(encoding="utf-8"))["survivors"])
           if BASELINE.exists() else set())
    if write and partial:
        print("!! 拒绝: --since/--changed 只扫改动行, 用它写基线会抹掉未扫到的已知条目。"
              "\n   基线请用全量扫描更新: python3 mutate_auto.py --baseline")
        return 1
    if write:
        # 只替换跑过模块的条目, 其余模块的既有条目原样保留
        merged = sorted([s for s in old if s.split("|")[0] not in ran] + now)
        BASELINE.write_text(json.dumps({"survivors": merged}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"基线已更新 {BASELINE.name}: 本次模块 {len(now)} 条, 全库共 {len(merged)} 条")
        return 0
    if not BASELINE.exists():
        print("!! 无基线文件, 先跑 --baseline 生成")
        return 1
    known = {s for s in old if s.split("|")[0] in ran}   # 只与跑过的模块比
    new = [s for s in now if s not in known]
    fixed = [] if partial else [s for s in sorted(known) if s not in now]
    for s in new:
        print(f"  新增存活(测试缺口): {s}")
    for s in fixed:
        print(f"  已修补(可更新基线): {s}")
    if new:
        print(f"\n失败: {len(new)} 个新增存活变异 —— 新代码缺少能区分它的测试")
        return 1
    print(f"\n通过: 无新增存活(比对模块 {ran}: 基线 {len(known)} 个, 本次 {len(now)} 个)")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a}
    flags = {a for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
    mods = args or ["claim_eval.py", "answer_match.py", "rubric_eval.py",
                    "eval_task.py", "paired_bench.py"]
    since = opts.get("--since") or ("HEAD" if "--changed" in flags else None)
    res = run(mods, limit=int(opts.get("--limit", 0)) or None,
              seed=int(opts.get("--seed", 0)), since=since)
    if flags & {"--baseline", "--check"}:
        sys.exit(check_against_baseline(res, write="--baseline" in flags,
                                        partial=since is not None))
