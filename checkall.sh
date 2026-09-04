#!/bin/sh
# 发布前全量验证: 按 快->慢 顺序跑完所有检查层, 汇总后统一退出。
# 本脚本是发布流程的可执行文档 —— 各层的分层理由见对应工具头部, 这里只保证
# 顺序正确、全部跑到、任一失败都反映在退出码里(不因前面失败就跳过后面, 一次看全)。
#
#   1 快速套件      ~5s    177 测试 + 结构自检(pre-commit 用的就是它)
#   2 钩子集成      ~26s   建临时仓库+裸远端跑真钩子(pre-commit/pre-push 行为)
#   3 功效校准      ~5s    required_pairs 与教科书功效公式对量级(真跑蒙特卡洛)
#   4 手挑变异      ~81s   42 个高阶语义变异 + 腐坏检测(PATTERN-MISS)
#   5 机械变异棘轮  ~25min 473 个变异点全量(每点跑一次 3 秒套件), 与基线比对, 新增存活即失败
#
# 刻意不在此脚本内的第六层: reproduce_findings.py, 需注入真实模型调用。三项检查:
#   1 脚手架效应   ~32 次调用   固定模型, 严格脚手架 vs 裸指令
#   2 模型 null    ~32 次调用   两模型不可区分(需第二个模型)
#   3 推算增量     ~250 次调用  严格提示上自检仍有增量(需 judge 做 claim 抽取+grounding)
# 它检的是"记录的实测结论是否漂移", 依赖外部服务与配额, 不能进无网可跑的清单;
# 三项的判据逻辑都已被 test_reproduce_findings.py 用假 call/judge 覆盖(在第 1 层里)。
# 发布前若涉及结论变更, 手动跑:
#   python3 -c "import reproduce_findings as r; r.main(call, call_b=..., judge=...)"
#
# 用法: sh checkall.sh          全部五层
#       sh checkall.sh --fast   只跑前四层(跳过约 25 分钟的棘轮)
cd "$(dirname "$0")"
fail=0
stage() {
    name="$1"; shift
    printf '\n=== %s ===\n' "$name"
    if "$@"; then
        echo "PASS  $name"
    else
        echo "FAIL  $name"
        fail=1
    fi
}

stage "1 快速套件" sh runtests.sh
stage "2 钩子集成" python3 -B check_hooks.py
stage "3 功效校准" python3 -B -c \
    "import sys; sys.path.insert(0, 'tests'); import test_claim_eval as t; \
     t.calibrate_required_pairs_against_formula(); print('规划器与功效公式吻合')"
stage "4 手挑变异" sh mutate.sh
if [ "$1" != "--fast" ]; then
    stage "5 机械变异棘轮" python3 -B mutate_auto.py --check
else
    printf '\n=== 5 机械变异棘轮 ===\nSKIP  (--fast)\n'
fi

printf '\n=== 汇总 ===\n'
[ "$fail" -eq 0 ] && echo "全部检查通过" || echo "存在失败层, 见上文 FAIL 行"
exit $fail
