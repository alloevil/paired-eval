#!/bin/sh
# claim-eval 全套件入口。任何套件失败即非零退出 —— pre-commit 钩子与 CI 共用。
# 结构自检: 定义的测试数必须等于实际执行数(防"定义在 __main__ 块后被静默跳过"),
# __main__ 块必须恰好一个(防双块重复/漏收集)。两个陷阱都有真实踩坑记录。
# python3 -B: 不读写字节码缓存。pyc 只按 (源mtime秒级, 源size) 验缓存 ——
# 同大小的改动若发生在同一秒内, 验证器看不见, 门禁会用过期字节码给出假绿。
# 本仓库做变异测试时真实撞到过一次(== 改 != 不变大小 -> 变异被报告为"存活")。
# 末行汇总是刻意的: 逐文件输出被 `| tail -3` 之类截断时, 靠前的 FAILED 会看不见 ——
# 本仓库真实踩过(第115轮: 两个套件已红, 我看 tail -3 全绿就继续往下做)。
# 汇总行永远在最后, 故任何 tail 都能看到成败。
cd "$(dirname "$0")"
fail=0
files=0
bad=""
for f in test_*.py; do
    defs=$(grep -c "^def test_" "$f")
    mains=$(grep -c "^if __name__" "$f")
    if [ "$mains" -ne 1 ]; then
        echo "[$f] FAILED: __main__ 块数量为 $mains, 必须恰好 1"
        fail=1; bad="$bad $f"
        continue
    fi
    if out=$(python3 -B "$f" 2>&1); then
        n=$(echo "$out" | tail -1 | grep -o '^[0-9][0-9]*')
        if [ "${n:-0}" -ne "$defs" ]; then
            echo "[$f] FAILED: 定义了 $defs 个测试, 只执行了 ${n:-0} 个 —— 有测试被静默跳过"
            fail=1; bad="$bad $f"
            continue
        fi
        echo "[$f] $n tests passed"
        files=$((files + n))
    else
        echo "[$f] FAILED"
        echo "$out" | tail -5
        fail=1; bad="$bad $f"
    fi
done
if [ "$fail" -ne 0 ]; then
    echo "=== 套件失败:$bad ==="
else
    echo "=== 全部通过: $files 测试 ==="
fi
exit $fail
