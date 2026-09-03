#!/bin/sh
# 变异测试: 检验测试套件自身的杀伤力。每个变异是一处语义改动, 套件必须失败(killed)。
# "测试全绿"只说明没触发失败, 不等于测试能抓到bug —— 杀伤率才是套件质量的度量。
# 用法: sh mutate.sh   (每次变异后自动还原源文件)
# 定向变异测试: 每个变异都是一处语义改动, 套件必须失败(killed) 才算有杀伤力
cd ~/claim-eval
run() {
  desc="$1"; file="$2"; old="$3"; new="$4"
  cp "$file" /tmp/mut.bak
  python3 - "$file" "$old" "$new" <<'PY'
import sys, pathlib
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
t = pathlib.Path(p).read_text(encoding="utf-8")
assert old in t, f"变异模式未命中: {old!r}"
pathlib.Path(p).write_text(t.replace(old, new, 1), encoding="utf-8")
PY
  if sh runtests.sh >/dev/null 2>&1; then echo "SURVIVED  $desc"; else echo "killed    $desc"; fi
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
