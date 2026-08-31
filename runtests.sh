#!/bin/sh
# claim-eval 全套件入口。任何套件失败即非零退出 —— pre-commit 钩子与 CI 共用。
set -e
cd "$(dirname "$0")"
fail=0
for f in test_*.py; do
    if out=$(python3 "$f" 2>&1); then
        echo "$out" | tail -1 | sed "s/^/[$f] /"
    else
        echo "[$f] FAILED"
        echo "$out" | tail -5
        fail=1
    fi
done
exit $fail
