#!/bin/sh
# 手挑语义变异: 检验测试套件的杀伤力。每个变异是一处语义改动, 套件必须失败(killed)。
# "测试全绿"只说明没触发失败, 不等于测试能抓到bug —— 杀伤率才是套件质量的度量。
#
# 与 mutate_auto.py 的分工(两者互补, 不重复):
#   本脚本    整表达式替换等高阶变异(加权退化为未加权、双侧检验改单侧、复核判定反转),
#             AST 工具生成不出来; 代价是靠字符串匹配定位, 代码改形状就会 PATTERN-MISS。
#   自动工具  机械枚举比较符/布尔/常量, 覆盖面完整无选择偏差, 带基线棘轮与 --changed。
#
# 退出码: 任何 SURVIVED 或 PATTERN-MISS 都非零 —— PATTERN-MISS 是腐坏信号:
# 变异模式失配说明被测代码换了形状, 必须复查新形状是否仍被测试覆盖。
# 用法: sh mutate.sh   (每次变异后自动还原源文件)
cd "$(dirname "$0")"
fail=0
run() {
  desc="$1"; file="$2"; old="$3"; new="$4"
  cp "$file" /tmp/mut.bak
  if ! python3 -B - "$file" "$old" "$new" <<'PY'
import sys, pathlib
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
t = pathlib.Path(p).read_text(encoding="utf-8")
assert old in t, f"变异模式未命中: {old!r}"
pathlib.Path(p).write_text(t.replace(old, new, 1), encoding="utf-8")
PY
  then
    echo "PATTERN-MISS  $desc (代码已改形状, 复查覆盖)"
    cp /tmp/mut.bak "$file"
    fail=1
    return
  fi
  if sh runtests.sh >/dev/null 2>&1; then
    echo "SURVIVED  $desc"
    fail=1
  else
    echo "killed    $desc"
  fi
  cp /tmp/mut.bak "$file"
}
run "wilson_ci: z 1.96->1.0"            claim_eval.py "def wilson_ci(k, n, z=1.96)" "def wilson_ci(k, n, z=1.0)"
run "mcnemar: 双侧->单侧"                claim_eval.py "return min(1.0, 2 * tail)" "return min(1.0, tail)"
run "holm: (n-rank)->n (退化Bonferroni)" claim_eval.py "(n - rank) * pvalues[i]" "n * pvalues[i]"
run "pass_hat_k: s<k 返回1.0"            claim_eval.py "if successes < k:
        return 0.0" "if successes < k:
        return 1.0"
run "aggregate_world: 加权退化为未加权"   claim_eval.py '"weighted_precision": wp(sup) / wdenom if wdenom else None' '"weighted_precision": len(sup) / denom if denom else None'
run "run_paired: 取消顺序交替"            paired_bench.py "a_first = idx % 2 == 0" "a_first = True"
run "corroborate: 复核判定反转"           claim_eval.py 'if v2["verdict"] == "contradicted":' 'if v2["verdict"] != "contradicted":'
run "rubric: 分母漏掉 not_met"            rubric_eval.py "denom = met_w + not_w" "denom = met_w"
run "parse_number: 百分号乘100"           answer_match.py "return value / 100.0" "return value * 100.0"
run "trajectory: 加权退化"                claim_eval.py "weighted_grounding_rate=wsum(\"grounded\") / total_w" "weighted_grounding_rate=count(\"grounded\") / n"
run "wilson: 下界钳制去掉"        claim_eval.py "return (max(0.0, center - half), min(1.0, center + half))" "return (center - half, min(1.0, center + half))"
run "wilson: 上界钳制去掉"        claim_eval.py "return (max(0.0, center - half), min(1.0, center + half))" "return (max(0.0, center - half), center + half)"
run "mcnemar: n==0 返回0"         claim_eval.py "if n == 0:
        return 1.0" "if n == 0:
        return 0.0"
run "holm: 去掉单调强制"          claim_eval.py "running = max(running, min(1.0, (n - rank) * pvalues[i]))" "running = min(1.0, (n - rank) * pvalues[i])"
run "divergence: >= 改 >"         paired_bench.py 'row["divergent"] = row["spread"] >= divergence' 'row["divergent"] = row["spread"] > divergence'
run "rotation: shift 方向反转"     paired_bench.py "order = names[shift:] + names[:shift]" "order = names[:shift] + names[shift:]"
run "derived: 容差方向反转"        claim_eval.py "ok = abs(value - claimed) <= rel_tol * max(abs(value), abs(claimed), 1e-9)" "ok = abs(value - claimed) >= rel_tol * max(abs(value), abs(claimed), 1e-9)"
run "safe_verdict: 无fallback也降级" claim_eval.py "if fallback is not None:" "if fallback is None:"
run "boxed: depth 判定改错"        answer_match.py "if depth == 0:
                return text[start:j], i" "if depth >= 0:
                return text[start:j], i"
run "grade: no_slot/blank 反转"    answer_match.py 'reason = "blank" if not str(response).strip() else "no_slot"' 'reason = "no_slot" if not str(response).strip() else "blank"'
run "meter: chars_in 少算system"   claim_eval.py "self.llm_chars_in += len(prompt) + len(system)" "self.llm_chars_in += len(prompt)"
run "throttled_pmap: 保序丢失"     claim_eval.py "return list(ex.map(fn, xs))" "return list(reversed(list(ex.map(fn, xs))))"
run "saturation: 恒败归入有信息"   paired_bench.py "elif runs and not any(runs):
            sat_fail.append(tid)" "elif False:
            sat_fail.append(tid)"
run "screen: 跳过复核阶段"         paired_bench.py "if confirm_n is None or not flagged:" "if True:"
run "指纹: 截断长度 12->8"          eval_task.py 'hexdigest()[:12]' 'hexdigest()[:8]'
run "指纹: 少算 rubric prompt"      eval_task.py "re_.JUDGE_RUBRIC_SYSTEM)" ")"
run "summarize: 混指纹检查失效"      eval_task.py "if len(fps) > 1 and strict_fp:" "if len(fps) > 2 and strict_fp:"
run "summarize: 窗口比较 > 改 >="    eval_task.py "if span > max_span_s:" "if span >= max_span_s:"
run "repeat: stdev 门槛 >1 改 >0"    eval_task.py "statistics.stdev(scores) if len(scores) > 1 else None" "statistics.stdev(scores) if len(scores) > 0 else None"
run "evaluate_pair: 取消先后交换"    eval_task.py "a_first = rep % 2 == 0
" "a_first = True
"
run "repeated: 逐轮交替取消"         paired_bench.py "a_first = rep % 2 == 0" "a_first = True"
run "discordant: a_only 条件反转"    paired_bench.py "ta += bool(xa) and not bool(xb)" "ta += bool(xb) and not bool(xa)"
run "pairwise: 只比第一对"           paired_bench.py "for j in range(i + 1, len(names))]" "for j in range(i + 1, min(i + 2, len(names)))]"
run "concentration: max 改 sum"      paired_bench.py 'concentration = (max(r["a_only"] + r["b_only"] for r in by_task) / total_disc' 'concentration = (sum(r["a_only"] + r["b_only"] for r in by_task) / total_disc'
run "saturation: all 改 any"         paired_bench.py "if runs and all(runs):" "if runs and any(runs):"
run "traj门禁: 命中判定放宽"          claim_eval.py 'hit = [r for r in got if r["verdict"] == p["expect"]]' 'hit = [r for r in got if r["verdict"] != "grounded"]'
run "selfcheck: 干净文档分支失效"     claim_eval.py "if not planted:" "if False:"
run "extract: 不过滤 verifiable"     claim_eval.py "return [c for c in _extract_all(text, llm) if c[\"verifiable\"]]" "return _extract_all(text, llm)"
run "canary: 分离度门槛失效"          rubric_eval.py "separation >= margin" "separation >= -1"
run "match_set: 去重失效"             answer_match.py "if j not in matched_gold and eq(p, g):" "if eq(p, g):"
run "match_numeric: abs_tol 覆盖rel"  answer_match.py "abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)" "abs(a - b) <= min(rel_tol * max(abs(a), abs(b)), abs_tol)"
run "aggregate_grades: attempted口径" answer_match.py 'att = sum(g["verdict"] != "not_attempted" for g in grades)' 'att = len(grades)'

exit $fail
