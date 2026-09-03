#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""系统化变异测试: AST 遍历机械生成变异, 消除手挑变异的选择偏差。

手挑变异只能测到"我想到的地方"(本仓库前三批: 自选10个0存活, 故意找茬则7个存活)。
本工具枚举所有可变异节点(比较符/布尔运算/数值常量/not), 逐个应用并跑套件。

必做对照: 先验证 ast.unparse 空变异后套件仍全绿 —— 否则 unparse 自身的改写
(丢注释、改引号、规范化)会让所有变异都"被杀死", 得出虚假的满分。

用法: python3 mutate_auto.py [模块...] [--limit N] [--seed S]


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
这类结论应记录而非补测: 为它们写的测试只会把实现细节钉死, 让合理重构无谓失败。
"""
import ast
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


class _Mutator(ast.NodeTransformer):
    """把第 target 个可变异点改掉; 其余原样。"""

    def __init__(self, target, skip_ids=frozenset()):
        self.target = target
        self.skip_ids = skip_ids
        self.seen = 0
        self.applied = None

    def _hit(self, desc, lineno):
        if self.seen == self.target:
            self.applied = f"L{lineno}: {desc}"
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
    m = _Mutator(-1, skip_ids)
    m.visit(tree)
    return m.seen


def suite_passes():
    return subprocess.run(["sh", str(ROOT / "runtests.sh")],
                          capture_output=True).returncode == 0


def run(modules, limit=None, seed=0):
    results = []
    for mod in modules:
        path = ROOT / mod
        original = path.read_text(encoding="utf-8")
        tree = ast.parse(original)
        # 对照: 空变异(仅 unparse)必须仍然全绿
        path.write_text(ast.unparse(tree), encoding="utf-8")
        baseline_ok = suite_passes()
        path.write_text(original, encoding="utf-8")
        if not baseline_ok:
            print(f"!! {mod}: unparse 空变异即失败, 该模块的变异结果不可信(跳过)")
            continue
        tree_c = ast.parse(original)          # skip_ids 依赖节点对象同一性: 必须同一棵树
        total = count_points(tree_c, _noise_constants(tree_c))
        idxs = list(range(total))
        if limit and limit < total:
            idxs = sorted(random.Random(seed).sample(idxs, limit))
        print(f"== {mod}: {total} 个可变异点(已滤除默认参数/切片长度等已知等价类), "
              f"本次跑 {len(idxs)} 个 (unparse 对照通过)")
        for i in idxs:
            tree_i = ast.parse(original)
            mut = _Mutator(i, _noise_constants(tree_i))
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
            path.write_text(original, encoding="utf-8")
            results.append((mod, mut.applied, killed))
            if not killed:
                print(f"  SURVIVED  {mod} {mut.applied}")
    killed = sum(1 for _, _, k in results if k)
    print(f"\n杀伤率: {killed}/{len(results)}")
    for mod, desc, k in results:
        if not k:
            print(f"  存活: {mod} {desc}")
    return results


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a}
    mods = args or ["claim_eval.py", "answer_match.py", "rubric_eval.py",
                    "eval_task.py", "paired_bench.py"]
    run(mods, limit=int(opts.get("--limit", 0)) or None, seed=int(opts.get("--seed", 0)))
