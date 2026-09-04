# -*- coding: utf-8 -*-
"""claim_eval 离线结构测试:mock llm/search,零网络零API,验证纯代码路径。
运行: python3 test_claim_eval.py
"""
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 项目根: 让 `python3 tests/x.py` 直接可跑
from paired_eval import claim_eval as ce

# ---------------------------------------------------------------- mocks

def mock_search(query):
    return f"evidence({query})"

def _claim(text, quote, importance, query):
    return {"text": text, "source_quote": quote, "verifiable": True,
            "importance": importance, "search_query": query}



def mock_llm(prompt, system, schema):
    # 抽取器: 按输入文本返回脚本化 claims
    if system is ce.EXTRACT_SYSTEM:
        if "CLEAN_DOC" in prompt:
            return {"claims": [
                {"text": "E干净事实", "source_quote": "E", "verifiable": True,
                 "importance": "core", "search_query": "qE"}]}
        if "POISON_DOC" in prompt:  # 干净文档但judge会误判 → 测误报计数
            return {"claims": [
                {"text": "F真事实但被误判", "source_quote": "F", "verifiable": True,
                 "importance": "core", "search_query": "qF"}]}
        return {"claims": [
            {"text": "A核心事实", "source_quote": "A", "verifiable": True,
             "importance": "core", "search_query": "qA"},
            {"text": "B植入错误", "source_quote": "B错", "verifiable": True,
             "importance": "detail", "search_query": "qB"},
            {"text": "C观点句", "source_quote": "C", "verifiable": False,
             "importance": "detail", "search_query": "qC"},
            {"text": "D需重试的事实", "source_quote": "D", "verifiable": True,
             "importance": "supporting", "search_query": "qD"}]}
    # 开放世界判定器: A/E支持, B矛盾(两轮都矛盾), F误判矛盾, D首轮不足重构后支持,
    # S单源矛盾: 首轮矛盾、复核轮(corr-q)支持
    if system is ce.JUDGE_WORLD_SYSTEM:
        pick = lambda v: {"verdict": v, "evidence_quote": "q", "reasoning": "r"}
        if "S单源矛盾" in prompt:
            return pick("supported" if "corr-q" in prompt else "contradicted")
        if "B植入错误" in prompt or "F真事实" in prompt:
            return pick("contradicted")
        if "D需重试" in prompt:
            return pick("supported" if "reformed" in prompt else "insufficient")
        return pick("supported")
    if system is ce.REFORMULATE_SYSTEM:
        return {"query": "reformed"}
    if system is ce.CORROBORATE_SYSTEM:
        return {"query": "corr-q"}
    # 轨迹判定器: 按claim关键词
    if system is ce.JUDGE_TRAJ_SYSTEM:
        if "有据" in prompt:
            return {"verdict": "grounded", "source_tool_call": "tc_1",
                    "evidence_quote": "q", "reasoning": "r"}
        if "夸大" in prompt:
            return {"verdict": "distorted", "source_tool_call": "tc_1",
                    "evidence_quote": "q", "reasoning": "r"}
        return {"verdict": "fabricated", "source_tool_call": None,
                "evidence_quote": "", "reasoning": "r"}
    # 数值推导审计器: 按claim关键词给算式;evil用于测白名单拦截
    if system is ce.DERIVED_SYSTEM:
        if "对的" in prompt:
            return {"computable": True, "expression": "10 - 4", "claimed_value": 6,
                    "inputs": [{"value": 10, "tool_call_id": "tc_1"}]}
        if "错的" in prompt:
            return {"computable": True, "expression": "10 - 4", "claimed_value": 5,
                    "inputs": [{"value": 10, "tool_call_id": "tc_1"}]}
        if "恶意" in prompt:
            return {"computable": True, "expression": "__import__('os').getcwd()",
                    "claimed_value": 1, "inputs": []}
        if "语法炸弹" in prompt:  # 字符集合法但语法非法
            return {"computable": True, "expression": "1 ++", "claimed_value": 1, "inputs": []}
        if "除零" in prompt:
            return {"computable": True, "expression": "1/0", "claimed_value": 1, "inputs": []}
        return {"computable": False}
    raise AssertionError(f"未知system: {system[:20]}")


# ---------------------------------------------------------------- tests

def test_run_world_and_retry():
    out = ce.run_world("DOC", mock_llm, mock_search)
    m, claims = out["metrics"], out["claims"]
    assert m["n"] == 3, "verifiable=False 的 C 应被过滤"
    assert m["unverifiable"] == 1 and abs(m["unverifiable_rate"] - 0.25) < 1e-9, \
        "观点句不进分母但必须单独计量"
    assert m["supported"] == 2 and m["contradicted"] == 1 and m["insufficient"] == 0
    assert abs(m["precision"] - 2 / 3) < 1e-9
    # 加权: supported=A(core,3)+D(supporting,2)=5, contradicted=B(detail,1) → 5/6
    assert abs(m["weighted_precision"] - 5 / 6) < 1e-9
    d = next(c for c in claims if c["text"] == "D需重试的事实")
    assert d["retries"] == 1 and d["verdict"] == "supported", "insufficient 应重构query后翻案"


def test_retry_exhaustion():
    claim = {"text": "D需重试的事实", "source_quote": "D", "verifiable": True,
             "importance": "supporting", "search_query": "qD"}
    r = ce.verify_world(claim, mock_llm, mock_search, max_retries=0)
    assert r["verdict"] == "insufficient" and r["retries"] == 0, "重试耗尽应保留 insufficient"


def test_selfcheck_recall_and_fp():
    cases = [
        {"text": "DOC", "planted": [{"substr": "B错", "desc": "植入B"},
                                    {"substr": "不存在片段", "desc": "永远漏检"}]},
        {"text": "CLEAN_DOC", "planted": []},
        {"text": "POISON_DOC", "planted": []},
    ]
    g = ce.selfcheck(cases, mock_llm, mock_search)
    assert g["planted"] == 2 and g["caught"] == 1 and g["recall"] == 0.5
    assert g["clean_claims"] == 2 and g["false_positives"] == 1 and g["fp_rate"] == 0.5
    assert g["fp_details"][0]["claim"] == "F真事实但被误判"


def test_trajectory():
    obs = [{"tool_call_id": "tc_1", "tool": "t", "observation": "o"}]
    out = ce.run_trajectory(["这条有据", "这条夸大", "这条无中生有"], obs, mock_llm)
    m = out["metrics"]
    assert m["grounding_rate"] == m["distortion_rate"] == m["fabrication_rate"] == 1 / 3
    assert out["claims"][2]["source_tool_call"] is None


def test_derived_branches_and_guard():
    obs = [{"tool_call_id": "tc_1", "observation": "x=10, y=4"}]
    assert ce.verify_derived("对的差值", obs, mock_llm)["verdict"] == "derived-ok"
    assert ce.verify_derived("错的差值", obs, mock_llm)["verdict"] == "derived-wrong"
    assert ce.verify_derived("没输入的量", obs, mock_llm)["verdict"] == "not-derivable"
    r = ce.verify_derived("恶意表达式", obs, mock_llm)
    assert r["verdict"] == "bad-expression", "非算术表达式必须被白名单拦截,禁止进eval"
    r_syn = ce.verify_derived("语法炸弹", obs, mock_llm)
    assert r_syn["verdict"] == "bad-expression" and "SyntaxError" in r_syn["error"], \
        "字符集合法但语法非法: 必须判bad-expression而不是崩掉整批"
    r_div = ce.verify_derived("除零", obs, mock_llm)
    assert r_div["verdict"] == "bad-expression" and "ZeroDivisionError" in r_div["error"]


def test_tolerance_boundary():
    obs = [{"tool_call_id": "tc_1", "observation": "x=10, y=4"}]
    # claimed=6, recomputed=6, rel_tol边界: |6-6.13|>2%*6 → wrong; 5.9在2%内 → ok
    r_ok = ce.verify_derived("对的差值", obs, mock_llm, rel_tol=0.02)
    assert r_ok["verdict"] == "derived-ok"
    r_edge = ce.verify_derived("错的差值", obs, mock_llm, rel_tol=0.2)
    assert r_edge["verdict"] == "derived-ok", "容差放宽到20%时 5 vs 6 应通过"


def test_wilson_ci():
    assert ce.wilson_ci(0, 0) is None
    lo, hi = ce.wilson_ci(2, 3)
    assert 0.0 <= lo < 2 / 3 < hi <= 1.0, "区间必须含点估计"
    lo2, hi2 = ce.wilson_ci(20, 30)
    assert (hi2 - lo2) < (hi - lo), "同比例下样本更大区间必须更窄"
    m = ce.run_world("DOC", mock_llm, mock_search)["metrics"]
    assert m["precision_ci"][0] < m["precision"] < m["precision_ci"][1]
    g = ce.selfcheck([{"text": "DOC", "planted": [{"substr": "B错", "desc": "b"}]},
                      {"text": "CLEAN_DOC", "planted": []}], mock_llm, mock_search)
    assert g["recall_ci"] is not None and g["fp_rate_ci"] is not None
    try:
        ce.wilson_ci(5, 3)
        assert False, "k>n 必须拒绝而不是返回胡话区间"
    except ValueError:
        pass


def test_make_resilient():
    calls, naps = [], []
    def flaky(prompt, system, schema):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("stream timeout")
        return {"ok": True}
    r = ce.make_resilient(flaky, tries=3, sleep=naps.append)
    assert r("p", "s", {}) == {"ok": True}
    assert len(calls) == 3 and naps == [2.0, 4.0], "线性退避: 2s, 4s"
    def dead(prompt, system, schema):
        raise RuntimeError("permanent")
    try:
        ce.make_resilient(dead, tries=2, sleep=lambda _: None)("p", "s", {})
        assert False, "耗尽重试必须抛出"
    except RuntimeError:
        pass


def test_throttled_pmap():
    pmap = ce.throttled_pmap(max_workers=2)
    assert pmap(lambda x: x * x, range(6)) == [0, 1, 4, 9, 16, 25], "结果必须保序"


def test_paired_compare():
    # 完全相同 → 全平局, p≈1, CI 含 0
    r = ce.paired_compare([1, 0, 1, 0] * 5, [1, 0, 1, 0] * 5)
    assert r["mean_diff"] == 0 and r["ties"] == 20
    assert r["p_value"] > 0.9 and r["diff_ci"][0] <= 0 <= r["diff_ci"][1]
    # A 全面胜出 20 任务 → 显著, CI 排除 0
    r2 = ce.paired_compare([1] * 20, [0] * 20)
    assert r2["wins"] == 20 and r2["p_value"] < 0.01 and r2["diff_ci"][0] > 0
    # 同 seed 完全确定
    a, b = [0.9, 0.7, 0.8, 0.6, 0.85], [0.7, 0.75, 0.6, 0.65, 0.7]
    assert ce.paired_compare(a, b) == ce.paired_compare(a, b)
    # n=3 全胜也不许显著 —— 小样本防过度自信(符号翻转下限 p=2/8)
    r3 = ce.paired_compare([1, 1, 1], [0, 0, 0])
    assert r3["p_value"] > 0.05
    # 配对约束: 不等长必须拒绝
    try:
        ce.paired_compare([1], [1, 0])
        assert False, "不等长应抛 ValueError"
    except ValueError:
        pass


def test_required_tasks():
    # 效应越大所需任务越少(单调性)
    n_big = ce.required_tasks(0.5, 0.05, sims=200)
    n_small = ce.required_tasks(0.25, 0.15, sims=200)
    assert n_big and n_small and n_big < n_small
    # 上一轮demo场景(15任务 A胜3负1): 需要的任务数远超15
    assert ce.required_tasks(0.2, 0.067, sims=200) > 15
    # 微弱效应在小n_max内达不到功效 → None
    assert ce.required_tasks(0.21, 0.20, sims=100, n_max=64) is None
    # 同参数完全确定
    assert ce.required_tasks(0.4, 0.1, sims=100) == ce.required_tasks(0.4, 0.1, sims=100)
    # 参数校验: p_win 必须大于 p_loss
    try:
        ce.required_tasks(0.2, 0.3)
        assert False, "p_loss>=p_win 应抛 ValueError"
    except ValueError:
        pass


def test_corroboration():
    claim_s = _claim("S单源矛盾", "S", "core", "qS")
    claim_b = _claim("B植入错误", "B错", "detail", "qB")
    # 默认关闭: 单源即定罪(旧行为不变)
    r0 = ce.verify_world(claim_s, mock_llm, mock_search)
    assert r0["verdict"] == "contradicted" and "corroborated" not in r0
    # 开启: 复核轮不再矛盾 → 降级 insufficient, 标记未获复核
    r1 = ce.verify_world(claim_s, mock_llm, mock_search, corroborate=True)
    assert r1["verdict"] == "insufficient" and r1["corroborated"] is False
    assert "单源矛盾未获复核" in r1["reasoning"]
    # 真错误: 两轮独立检索都矛盾 → 维持 contradicted 并标记已复核
    r2 = ce.verify_world(claim_b, mock_llm, mock_search, corroborate=True)
    assert r2["verdict"] == "contradicted" and r2["corroborated"] is True


def test_meter():
    m = ce.Meter()
    ce.run_world("DOC", m.wrap_llm(mock_llm), m.wrap_search(mock_search))
    snap = m.snapshot()
    # llm: 抽取1 + judge(A,B,D不足,D重构后)4 + 重构1 = 6; search: qA,qB,qD,reformed = 4
    assert snap["llm_calls"] == 6 and snap["search_calls"] == 4
    assert snap["llm_chars_in"] > 0 and snap["llm_chars_out"] > 0
    d = ce.Meter.delta(snap, {"llm_calls": 1, "llm_chars_in": 0,
                              "llm_chars_out": 0, "search_calls": 0})
    assert d["llm_calls"] == 5 and d["search_calls"] == 4


def test_pass_hat_k():
    assert ce.pass_hat_k(8, 8, 3) == 1.0, "全过则任取k次必过"
    assert ce.pass_hat_k(0, 8, 2) == 0.0
    assert ce.pass_hat_k(4, 8, 1) == 0.5, "k=1 退化为 pass@1"
    assert ce.pass_hat_k(4, 8, 8) == 0.0, "successes<k 概率为0"
    assert abs(ce.pass_hat_k(4, 8, 2) - 3 / 14) < 1e-12, "C(4,2)/C(8,2)=6/28"
    ks = [ce.pass_hat_k(5, 8, k) for k in range(1, 6)]
    assert all(a >= b for a, b in zip(ks, ks[1:])), "k 增大可靠性单调不增"
    for bad in [(9, 8, 1), (-1, 8, 1), (4, 8, 0), (4, 8, 9)]:
        try:
            ce.pass_hat_k(*bad)
            assert False, f"非法参数应拒绝: {bad}"
        except ValueError:
            pass



def test_judge_insanity_guards():
    crazy = lambda p, s, sch: {"nonsense": True}
    claim = _claim("X", "X", "core", "qX")
    # world: 疯输出降级 insufficient(不进precision分母), 批次存活
    r = ce.verify_world(claim, crazy, mock_search, max_retries=0)
    assert r["verdict"] == "insufficient" and "非法" in r["reasoning"]
    # trajectory: 无中性桶, 必须响亮报错
    try:
        ce.verify_trajectory("X", [{"tool_call_id": "t", "observation": "o"}], crazy)
        assert False, "trajectory 疯输出应报错"
    except ValueError:
        pass
    # 抽取器: 整体畸形与单条畸形都响亮报错
    for bad in [crazy, lambda p, s, sch: {"claims": [{"text": "x", "verifiable": "yes"}]},
                lambda p, s, sch: {"claims": [{"text": "x", "verifiable": True,
                                               "importance": "huge", "search_query": "q"}]}]:
        try:
            ce.extract_claims("DOC", bad)
            assert False, "抽取器畸形输出应报错"
        except ValueError:
            pass
    # 重构query疯了: 沿用旧query继续, 不崩
    def half_crazy(p, s, sch):
        if s is ce.REFORMULATE_SYSTEM:
            return "garbage"
        return {"verdict": "insufficient", "evidence_quote": "", "reasoning": "r"}
    r2 = ce.verify_world(claim, half_crazy, mock_search, max_retries=1)
    assert r2["verdict"] == "insufficient" and r2["retries"] == 1
    # derived: claimed_value 缺失/非数 → bad-expression 而非 KeyError
    dm = lambda p, s, sch: {"computable": True, "expression": "1+1", "inputs": []}
    r3 = ce.verify_derived("X", [{"tool_call_id": "t", "observation": "o"}], dm)
    assert r3["verdict"] == "bad-expression" and "claimed_value" in r3["error"]



def test_statistical_calibration():
    """统计原语的自校准: 公式抄对了不算数,误差率要实测。种子固定,确定性复现。"""
    import random as _r
    rng = _r.Random(7)
    # Wilson CI 实测覆盖率必须落在标称 95% 附近
    for p_true, n in [(0.5, 40), (0.1, 60)]:
        trials, cover = 800, 0
        for _ in range(trials):
            k = sum(rng.random() < p_true for _ in range(n))
            lo, hi = ce.wilson_ci(k, n)
            cover += lo <= p_true <= hi
        rate = cover / trials
        assert 0.92 <= rate <= 0.99, f"覆盖率失准 p={p_true},n={n}: {rate:.3f}"
    # 置换检验在 H0(两系统同分布)下的一类错误率不得超过标称 alpha 太多
    sims, fp = 200, 0
    for i in range(sims):
        a = [1.0 if rng.random() < 0.6 else 0.0 for _ in range(30)]
        b = [1.0 if rng.random() < 0.6 else 0.0 for _ in range(30)]
        fp += ce.paired_compare(a, b, n_resamples=200, seed=i)["p_value"] < 0.05
    rate = fp / sims
    assert rate <= 0.09, f"H0 下假阳性率超标: {rate:.3f}(标称0.05,离散数据允许保守)"
    assert rate >= 0.005, f"假阳性率反常地低,检验可能失去功效: {rate:.3f}"



def test_power_cross_calibration():
    """规划器与检验器的交叉校准: required_tasks(正态近似)给出的 n,
    必须让 paired_compare(精确置换)在真实效应下达到接近标称的功效 ——
    否则用户按虚低的 n 建任务集,永远等不到显著。种子固定,确定性复现。"""
    import random as _r
    p_w, p_l = 0.5, 0.05
    n = ce.required_tasks(p_w, p_l, power=0.8, sims=300, seed=3)
    assert n is not None
    rng = _r.Random(11)
    sims, hits = 150, 0
    for i in range(sims):
        diffs = []
        for _ in range(n):
            r = rng.random()
            diffs.append(1.0 if r < p_w else (-1.0 if r < p_w + p_l else 0.0))
        res = ce.paired_compare(diffs, [0.0] * n, n_resamples=200, seed=i)
        hits += res["p_value"] < 0.05
    power = hits / sims
    assert power >= 0.70, f"规划器承诺80%功效,检验器实测仅 {power:.2f} —— n 被低估"



def test_meter_thread_safety():
    m = ce.Meter()
    wrapped = m.wrap_llm(lambda p, s, sch: {"ok": 1})
    wrapped_s = m.wrap_search(lambda q: "ev")
    pmap = ce.throttled_pmap(max_workers=8)
    pmap(lambda i: (wrapped("p", "s", {}), wrapped_s("q")), range(2000))
    snap = m.snapshot()
    assert snap["llm_calls"] == 2000 and snap["search_calls"] == 2000, \
        "并发下计数必须精确 —— 账单少记是静默错误"
    assert snap["llm_chars_in"] == 2000 * 2 and snap["llm_chars_out"] == 2000 * len(str({"ok": 1}))


def test_mcnemar_exact():
    """闭式核对: 双侧 = 2 * Bin(n,0.5) 下尾, 上限1。"""
    assert ce.mcnemar_exact(0, 0) == 1.0, "无不一致对 = 无方向证据"
    assert abs(ce.mcnemar_exact(0, 5) - 2 * (1 / 32)) < 1e-12       # 2*C(5,0)/2^5
    assert abs(ce.mcnemar_exact(5, 0) - 2 * (1 / 32)) < 1e-12, "对称"
    assert abs(ce.mcnemar_exact(1, 9) - 2 * (11 / 1024)) < 1e-12    # 2*(C(10,0)+C(10,1))/2^10
    assert ce.mcnemar_exact(3, 3) == 1.0, "对称不一致对: p 封顶1"
    assert ce.mcnemar_exact(0, 1) == 1.0, "单个不一致对不足以显著"
    # 效应越干净 p 越小; 一致对不参与(不作为参数存在)
    assert ce.mcnemar_exact(0, 20) < ce.mcnemar_exact(0, 5)
    try:
        ce.mcnemar_exact(-1, 2)
        assert False, "负计数应拒绝"
    except ValueError:
        pass

def test_holm_adjust():
    """闭式核对: 排序后 (n-rank)*p, 强制单调不减, 封顶1。"""
    assert ce.holm_adjust([]) == []
    assert ce.holm_adjust([0.02]) == [0.02], "单个检验不校正"
    # [0.01,0.04,0.03]: 0.01*3=0.03 -> 0.03*2=0.06 -> max(0.06,0.04*1)=0.06
    got = ce.holm_adjust([0.01, 0.04, 0.03])
    assert all(abs(g - e) < 1e-12 for g, e in zip(got, [0.03, 0.06, 0.06])), got
    assert ce.holm_adjust([0.5, 0.6]) == [1.0, 1.0], "封顶1"
    adj = ce.holm_adjust([0.001, 0.9, 0.9])
    assert adj[0] < 0.05 <= adj[1], "强效应经校正仍显著,弱的被压住"
    assert all(a >= b for a, b in zip(ce.holm_adjust([0.01, 0.02, 0.03]),
                                      [0.01, 0.02, 0.03])), "校正后不小于原值"
    try:
        ce.holm_adjust([0.5, 1.5])
        assert False, "越界p应拒绝"
    except ValueError:
        pass




def test_mcnemar_calibration():
    """McNemar 的实测一类错误率: H0(两系统同分布)下 p<0.05 的比例不得超标。
    离散检验偏保守是允许的, 反常地低则说明失去功效, 故设双侧界。种子固定。"""
    import random as _r
    rng = _r.Random(101)
    for p_succ, reps in [(0.5, 40), (0.8, 60)]:
        trials, fp = 400, 0
        for _ in range(trials):
            a_only = b_only = 0
            for _ in range(reps):
                xa, xb = rng.random() < p_succ, rng.random() < p_succ
                a_only += xa and not xb
                b_only += xb and not xa
            fp += ce.mcnemar_exact(a_only, b_only) < 0.05
        rate = fp / trials
        assert rate <= 0.075, f"一类错误超标 p={p_succ},reps={reps}: {rate:.3f}"
    # 功效抽查(冒烟级,非精确功效分析): 真实单侧优势应被高频检出。
    # 阈值 0.85 留出 MC 噪声余量 —— 实测该配置约 0.89, 卡在 0.90 是刀锋阈值。
    hits = 0
    for _ in range(200):
        a_only = b_only = 0
        for _ in range(20):
            xa, xb = rng.random() < 0.9, rng.random() < 0.4
            a_only += xa and not xb
            b_only += xb and not xa
        hits += ce.mcnemar_exact(a_only, b_only) < 0.05
    assert hits / 200 >= 0.85, f"强效应下功效过低: {hits/200:.2f}"


def test_holm_controls_fwer():
    """实测 family-wise error rate: 5个独立检验全在H0下, 原始p会频繁误报,
    Holm 校正后必须压到标称水平以内 —— 这就是多重比较校正的全部意义。"""
    import random as _r
    rng = _r.Random(202)
    trials, raw_fp, holm_fp = 2000, 0, 0
    for _ in range(trials):
        pvals = [rng.random() for _ in range(5)]   # H0 下 p ~ U(0,1)
        raw_fp += any(p < 0.05 for p in pvals)
        holm_fp += any(p < 0.05 for p in ce.holm_adjust(pvals))
    raw, holm = raw_fp / trials, holm_fp / trials
    assert 0.17 <= raw <= 0.28, f"未校正FWER应接近1-0.95^5≈0.226: {raw:.3f}"
    assert holm <= 0.06, f"Holm 校正后FWER应<=0.05(容MC噪声): {holm:.3f}"
    assert holm < raw / 3, f"校正必须显著改善: raw={raw:.3f} holm={holm:.3f}"


def test_planner_matches_exact_test():
    """规划器与检验器必须用同一把尺子: 在 required_tasks 给出的 n 上,
    mcnemar_exact 的实测功效不得低于目标。历史缺陷: 规划器用正态近似时
    这里只有 0.765(目标0.8), n 被系统性低估。"""
    import random as _r
    for p_w, p_l in [(0.5, 0.05), (0.3, 0.1)]:
        n = ce.required_tasks(p_w, p_l, power=0.8, sims=300, seed=5)
        assert n is not None
        rng = _r.Random(9)
        trials, hits = 300, 0
        for _ in range(trials):
            a_only = b_only = 0
            for _ in range(n):
                r = rng.random()
                if r < p_w:
                    a_only += 1
                elif r < p_w + p_l:
                    b_only += 1
            hits += ce.mcnemar_exact(a_only, b_only) < 0.05
        assert hits / trials >= 0.78, \
            f"规划n={n} 实测功效仅 {hits/trials:.3f}(目标0.8): 规划器低估了样本量"




def test_detectable_effect():
    """MDE 的性质: 随 n 单调下降, 随反向失误率上升, 与 required_tasks 互为反问题。"""
    mdes = [ce.detectable_effect(n, sims=200) for n in (28, 56, 112)]
    assert all(m is not None for m in mdes)
    assert mdes[0] > mdes[1] > mdes[2], f"n 越大 MDE 越小: {mdes}"
    assert ce.detectable_effect(28, p_loss=0.05, sims=200) > ce.detectable_effect(28, sims=200), \
        "存在反向失误时需要更大的胜率才能检出"
    # 反问题一致性: 用 MDE 反查所需 n, 应回到同一量级(容一个网格步)
    n = 56
    mde = ce.detectable_effect(n, sims=300)
    need = ce.required_tasks(mde, 0.001, sims=300)
    assert need is not None and need <= n * 1.3, f"MDE={mde} 反查 n={need} 与 {n} 不自洽"
    # 样本太小时任何效应都不可检出(不一致对不足以达到 alpha)
    assert ce.detectable_effect(3, sims=100) is None
    for bad in ((0,), (10, 1.0)):
        try:
            ce.detectable_effect(*bad)
            assert False, f"非法参数应拒绝: {bad}"
        except ValueError:
            pass



def test_trajectory_selfcheck():
    """轨迹检测器门禁: 忠实回答不得误报, 篡改回答必须按期望类别命中。"""
    obs = [{"tool_call_id": "tc_1", "tool": "s", "observation": "营收增长约4%"}]

    def traj_llm(prompt, system, schema):
        if system is ce.EXTRACT_SYSTEM:
            key = "夸大" if "夸大" in prompt else ("编造" if "编造" in prompt else "忠实")
            return {"claims": [{"text": f"{key}claim", "source_quote": "x", "verifiable": True,
                                "importance": "core", "search_query": "q"}]}
        if "夸大" in prompt:
            return {"verdict": "distorted", "source_tool_call": "tc_1",
                    "evidence_quote": "q", "reasoning": "r"}
        if "编造" in prompt:
            return {"verdict": "fabricated", "source_tool_call": None,
                    "evidence_quote": "", "reasoning": "r"}
        return {"verdict": "grounded", "source_tool_call": "tc_1",
                "evidence_quote": "q", "reasoning": "r"}

    g = ce.trajectory_selfcheck([{
        "observations": obs, "faithful": "忠实的总结",
        "planted": [{"response": "夸大的总结", "expect": "distorted", "desc": "数字夸大"},
                    {"response": "编造的总结", "expect": "fabricated", "desc": "无据补充"}],
    }], traj_llm)
    assert g["planted"] == 2 and g["caught"] == 2 and g["recall"] == 1.0
    assert g["clean_claims"] == 1 and g["false_positives"] == 0 and g["fp_rate"] == 0.0
    assert g["recall_ci"] is not None and g["fp_rate_ci"] is not None
    # 期望类别不符 = 漏检(抓到了但归错类,同样不合格)
    g2 = ce.trajectory_selfcheck([{
        "observations": obs, "planted": [
            {"response": "夸大的总结", "expect": "fabricated", "desc": "错配类别"}]}], traj_llm)
    assert g2["caught"] == 0, "判成distorted但期望fabricated: 记漏检"
    # 忠实回答被判非grounded = 误报
    fp_llm = lambda p, s, sch: (
        {"claims": [{"text": "c", "source_quote": "x", "verifiable": True,
                     "importance": "core", "search_query": "q"}]} if s is ce.EXTRACT_SYSTEM
        else {"verdict": "fabricated", "source_tool_call": None,
              "evidence_quote": "", "reasoning": "r"})
    g3 = ce.trajectory_selfcheck([{"observations": obs, "faithful": "忠实的总结"}], fp_llm)
    assert g3["false_positives"] == 1 and g3["fp_rate"] == 1.0


def test_unconfirmed_contradictions_surfaced():
    """复核降级的单源矛盾必须在指标层可见 —— 融进 insufficient 就等于信号消失。"""
    claim_s = _claim("S单源矛盾", "S", "core", "qS")
    claim_a = _claim("A核心事实", "A", "core", "qA")
    r_unconf = ce.verify_world(claim_s, mock_llm, mock_search, corroborate=True)
    r_ok = ce.verify_world(claim_a, mock_llm, mock_search)
    assert r_unconf["verdict"] == "insufficient" and r_unconf["corroborated"] is False
    m = ce.aggregate_world([r_unconf, r_ok])
    assert m["insufficient"] == 1 and m["unconfirmed_contradictions"] == 1, \
        "降级的矛盾既计入 insufficient 也单列"
    # 普通 insufficient(压根没证据)不该被算成未确认矛盾
    claim_d = _claim("D需重试的事实", "D", "supporting", "qD")
    r_plain = ce.verify_world(claim_d, mock_llm, mock_search, max_retries=0)
    m2 = ce.aggregate_world([r_plain])
    assert m2["insufficient"] == 1 and m2["unconfirmed_contradictions"] == 0, \
        "无证据 与 单源矛盾未确认 是两种不同诊断"



def test_trajectory_importance_weighting():
    """核心事实被编造 与 边角细节被编造 不该记同一笔账。
    丢掉 importance 会让重大幻觉被大量无害细节稀释。"""
    mkr = lambda verdict, imp: {"text": "t", "verdict": verdict, "importance": imp}
    # 情形A: 编造的是核心条(权重3), 3条grounded细节(各1) -> 未加权 3/4, 加权 3/6
    a = ce.aggregate_trajectory([mkr("fabricated", "core")] + [mkr("grounded", "detail")] * 3)
    assert a["grounding_rate"] == 0.75
    assert abs(a["weighted_grounding_rate"] - 0.5) < 1e-9
    assert abs(a["weighted_fabrication_rate"] - 0.5) < 1e-9, "核心幻觉占一半权重"
    # 情形B: 编造的是细节, grounded 是核心 -> 未加权同为 3/4, 加权大不同
    b = ce.aggregate_trajectory([mkr("fabricated", "detail")] + [mkr("grounded", "core")] * 3)
    assert b["grounding_rate"] == a["grounding_rate"], "未加权口径无法区分两种情形"
    assert abs(b["weighted_grounding_rate"] - 9 / 10) < 1e-9
    assert abs(b["weighted_fabrication_rate"] - 1 / 10) < 1e-9
    # 无 importance(独立 run_trajectory 只传文本)时加权字段为 None, 不报假数
    c = ce.aggregate_trajectory([{"text": "t", "verdict": "grounded"}])
    assert c["grounding_rate"] == 1.0 and c["weighted_grounding_rate"] is None
    # 非法 importance 同样降级为 None 而非崩溃
    d = ce.aggregate_trajectory([{"text": "t", "verdict": "grounded", "importance": "huge"}])
    assert d["weighted_grounding_rate"] is None


def test_planner_detail_mode():
    """规划器内部算出的实际功效不该被丢弃: 网格搜索会跳过临界点,
    余量常很大(实测 n=113 达 0.907), 只返回 n 会让调用者重做一遍蒙特卡洛。"""
    plain = ce.required_tasks(0.4, 0.05, sims=200, seed=3)
    det = ce.required_tasks(0.4, 0.05, sims=200, seed=3, detail=True)
    assert det["n"] == plain, "detail 模式不改变 n"
    assert det["power_target"] == 0.8 and det["alpha"] == 0.05
    assert det["achieved_power"] >= det["power_target"], "达标点的实际功效必须≥目标"
    assert 0.8 <= det["achieved_power"] <= 1.0
    # 不可达时 detail 仍给出结构化的 None, 便于记录"试过但达不到"
    unreach = ce.required_tasks(0.21, 0.20, sims=100, n_max=64, detail=True)
    assert unreach["n"] is None and unreach["achieved_power"] is None
    assert ce.required_tasks(0.21, 0.20, sims=100, n_max=64) is None, "非detail仍返回None"
    # MDE 同样
    mde_plain = ce.detectable_effect(56, sims=200, seed=4)
    mde_det = ce.detectable_effect(56, sims=200, seed=4, detail=True)
    assert mde_det["mde"] == mde_plain and mde_det["step"] == 0.01
    assert mde_det["achieved_power"] >= mde_det["power_target"]
    unreach2 = ce.detectable_effect(2, sims=100, detail=True)
    assert unreach2["mde"] is None and unreach2["achieved_power"] is None


def test_query_audit_trail():
    """检索词序列必须留痕: 只留最终 verdict 无法回答"这个结论是怎么来的"。"""
    # 一次判定就定案: 只有首轮 query
    r_ok = ce.verify_world(_claim("A核心事实", "A", "core", "qA"), mock_llm, mock_search)
    assert r_ok["queries"] == ["qA"] and "corroboration_verdict" not in r_ok
    # insufficient 触发重构: 记录首轮与重构后的词
    r_retry = ce.verify_world(_claim("D需重试的事实", "D", "supporting", "qD"),
                              mock_llm, mock_search, max_retries=1)
    assert r_retry["queries"] == ["qD", "reformed"], "重构后的检索词必须可见"
    assert r_retry["retries"] == 1
    # 复核: 追加复核词, 且复核轮的原始判定单独留痕
    r_corr = ce.verify_world(_claim("S单源矛盾", "S", "core", "qS"),
                             mock_llm, mock_search, corroborate=True)
    assert r_corr["queries"] == ["qS", "corr-q"]
    assert r_corr["corroboration_verdict"] == "supported", \
        "复核轮判了supported才导致降级, 这个原始判定不该只体现在文字里"
    assert r_corr["verdict"] == "insufficient" and r_corr["corroborated"] is False
    # 两轮都矛盾: 复核判定同样留痕
    r_conf = ce.verify_world(_claim("B植入错误", "B错", "detail", "qB"),
                             mock_llm, mock_search, corroborate=True)
    assert r_conf["corroboration_verdict"] == "contradicted" and r_conf["corroborated"] is True


def test_unverifiable_claims_exposed():
    """被排除的观点/hedge 句必须能被抽查: 只报数量, 就无法发现抽取器把
    难验证的事实句误标成观点句 —— 那是绕过评分最省力的路径。"""
    out = ce.run_world("DOC", mock_llm, mock_search)
    unv = out["unverifiable_claims"]
    assert [c["text"] for c in unv] == ["C观点句"], "被排除条目的原文必须带出"
    assert out["metrics"]["unverifiable"] == len(unv) == 1
    assert all(c["verifiable"] is False for c in unv)
    # 计分条目与被排除条目不重叠, 且两者之和等于抽取总数
    scored = [c["text"] for c in out["claims"]]
    assert "C观点句" not in scored and len(scored) + len(unv) == 4
    # 全部可验证时为空列表(不是None: "没有被排除"是明确事实)
    out2 = ce.run_world("CLEAN_DOC", mock_llm, mock_search)
    assert out2["unverifiable_claims"] == [] and out2["metrics"]["unverifiable"] == 0


def test_wilson_bounds_clamped():
    """概率区间不得越界, 哪怕越 1e-17: k=0/n=5 的未钳制下界是 -1.1e-17,
    k=n 的未钳制上界是 1.0000000000000002 —— 下游格式化/比较会被这种值坑到。
    (这两处钳制曾在变异测试中存活, 即无人断言过。)"""
    lo, hi = ce.wilson_ci(0, 5)
    assert lo == 0.0, f"下界必须精确钳到0, 得到 {lo!r}"
    lo2, hi2 = ce.wilson_ci(5, 5)
    assert hi2 == 1.0, f"上界必须精确钳到1, 得到 {hi2!r}"
    # 全域扫描: 任何 (k,n) 都必须落在 [0,1] 且有序
    for n in range(1, 12):
        for k in range(n + 1):
            a, b = ce.wilson_ci(k, n)
            assert 0.0 <= a <= b <= 1.0, f"k={k},n={n} 区间越界: ({a},{b})"


def test_extract_claims_filters_unverifiable():
    """extract_claims 的过滤此前从未被真正测过: run_world 自己用 _extract_all 过滤,
    而 trajectory 路由的 mock 全是 verifiable=True —— 去掉过滤在变异测试中存活。"""
    all_items = ce._extract_all("DOC", mock_llm)
    filtered = ce.extract_claims("DOC", mock_llm)
    assert len(all_items) == 4 and len(filtered) == 3
    assert all(c["verifiable"] for c in filtered)
    assert "C观点句" in [c["text"] for c in all_items]
    assert "C观点句" not in [c["text"] for c in filtered], "观点句必须被过滤掉"




def test_planner_nmax_boundary():
    """所需 n 恰好等于 n_max 时必须返回它(<= 语义), 而不是判为不可达。
    机械变异把 <= 改成 < 后存活: 此前用例的 n_max 从不与网格点重合。"""
    n = ce.required_tasks(0.9, 0.0, sims=200, n_max=8, seed=2)
    assert n == 8, f"n_max 恰好够用时应返回 8, 得到 {n!r}"
    # 略低于所需时仍返回 None
    assert ce.required_tasks(0.9, 0.0, sims=200, n_max=7, seed=2) is None


def test_mde_matches_exact_test():
    """MDE 的两个方向都要钉住(此前只钉了一半, 于是"计数器初值+1"的变异存活):
    - 充分性: 在报出的 MDE 上, 精确检验的实测功效必须达标(否则 MDE 被低估)。
    - 最小性: 在 MDE 往下三个网格步处, 功效必须低于目标(否则 MDE 被高估) ——
      MDE 的定义是"最小可检出", 只测充分性会让任何虚高的值都通过。
      实测该变异使 MDE 偏移 3-4 个网格步(0.19->0.22), 正是被最小性这一侧抓住。"""
    import random as _r
    n = 40

    def power_at(p, seed=23, trials=300):
        rng = _r.Random(seed)
        hits = 0
        for _ in range(trials):
            a_only = 0
            for _ in range(n):
                if rng.random() < p:
                    a_only += 1
            hits += ce.mcnemar_exact(a_only, 0) < 0.05
        return hits / trials

    mde = ce.detectable_effect(n, sims=300, seed=11)
    assert mde is not None
    assert power_at(mde) >= 0.75, f"MDE={mde} 实测功效不足: MDE 被低估"
    lower = round(mde - 3 * 0.01, 10)
    assert power_at(lower) < 0.80, \
        f"MDE={mde} 下移三步({lower})仍达标 {power_at(lower):.3f}: MDE 被高估, 不是最小值"




def test_derived_tolerance_inclusive():
    """derived 的容差同样是闭区间: 差值恰好等于容差应判 ok。
    (verify_derived 有独立的比较式, 与 match_numeric 那处需各自钉住。)"""
    obs = [{"tool_call_id": "tc_1", "observation": "x"}]
    # 重算=100, 声称=98, rel_tol=0.02 -> 容差 = 0.02*100 = 2.0, 差值恰好 2.0
    plan = lambda p, s, sch: {"computable": True, "expression": "100",
                              "claimed_value": 98, "inputs": []}
    assert ce.verify_derived("边界", obs, plan, rel_tol=0.02)["verdict"] == "derived-ok"
    plan2 = lambda p, s, sch: {"computable": True, "expression": "100",
                               "claimed_value": 97.9, "inputs": []}
    assert ce.verify_derived("略超", obs, plan2, rel_tol=0.02)["verdict"] == "derived-wrong"


def test_holm_never_exceeds_one():
    """校正后的 p 值必须封顶 1.0。此前用例里最小 p 乘完不超过 1,
    单调 running-max 把缺失的封顶掩盖了 —— 机械变异把 min(1.0,·) 改成 min(2.0,·) 后存活。"""
    adj = ce.holm_adjust([0.6, 0.7])      # 0.6*2 = 1.2, 未封顶就会溢出
    assert adj == [1.0, 1.0], f"必须封顶到1.0, 得到 {adj}"
    adj2 = ce.holm_adjust([0.4, 0.9, 0.95])   # 0.4*3 = 1.2
    assert all(p <= 1.0 for p in adj2) and adj2[0] == 1.0
    # 全域检查: 任意 p 组合的校正结果都不得越界
    for ps in ([0.99] * 5, [0.5, 0.51], [1.0], [0.0, 1.0], [0.34, 0.34, 0.34]):
        out = ce.holm_adjust(ps)
        assert all(0.0 <= p <= 1.0 for p in out), f"{ps} -> {out} 越界"




def test_derived_zero_guard():
    """容差分母里的 1e-9 只在两值都趋零时起作用: 它把容差压到近乎绝对零,
    否则 0 vs 0.01 这种"数量级级别的错"会被判 ok。机械变异把 1e-9 改成 ~1.0 后存活。"""
    obs = [{"tool_call_id": "tc_1", "observation": "x"}]
    mk = lambda expr, claimed: (lambda p, s, sch: {"computable": True, "expression": expr,
                                                   "claimed_value": claimed, "inputs": []})
    # 重算=0, 声称=0.01: 相对容差无意义, 必须判错(epsilon 不得放大容差)
    assert ce.verify_derived("零基准", obs, mk("0", 0.01), rel_tol=0.02)["verdict"] == "derived-wrong"
    # 两者都恰好为 0: 必须判 ok(epsilon 防止 0/0 退化)
    assert ce.verify_derived("双零", obs, mk("0", 0), rel_tol=0.02)["verdict"] == "derived-ok"




def test_planners_with_injected_rand():
    """注入确定性 rand 后, 规划器的边界可以被精确断言 ——
    蒙特卡洛函数的契约是统计性的, 随机性会吸收小扰动(变异测试中这两个函数
    存活率远高于其他代码), 注入是把它们变成可精确验证的唯一办法。"""
    # 恒 0: 每次抽样都 < p_win -> a_only=n, b_only=0 -> 强不一致 -> 功效=1
    always_win = lambda: 0.0
    d = ce.required_tasks(0.5, 0.1, sims=5, detail=True, rand=always_win)
    assert d["n"] == 8 and d["achieved_power"] == 1.0, \
        f"确定性全胜下最小网格点即达标: {d}"
    # 恒 1: 永不落入 p_win/p_loss 区间 -> 无不一致对 -> mcnemar p=1 -> 功效=0 -> 不可达
    never = lambda: 1.0
    assert ce.required_tasks(0.5, 0.1, sims=5, n_max=64, rand=never) is None
    assert ce.detectable_effect(20, sims=5, rand=never) is None
    # 恒 0 时 MDE 必须是第一个网格点(step), 因为它已达标
    mde = ce.detectable_effect(20, sims=5, step=0.05, rand=always_win)
    assert mde == 0.05, f"首个网格点即达标: {mde}"
    # achieved >= power 的相等边界: n=8 时每轮抽 8 次, sims=5 -> 共 40 次。
    # 前 4 轮(32次)全胜、第 5 轮(8次)全不胜 -> 功效恰为 4/5 = 0.8 = 默认 power。
    seq = iter([0.0] * 32 + [1.0] * 8)
    d2 = ce.required_tasks(0.5, 0.1, sims=5, n_max=8, detail=True,
                           rand=lambda: next(seq))
    assert d2["achieved_power"] == 0.8 and d2["n"] == 8, \
        f"功效恰好等于目标时必须判达标: {d2}"




def test_planner_grid_progression():
    """网格递进公式 max(n+4, int(n*1.4)) 必须被钉住: 起点8、增长率、下限步长
    三个常量此前都被随机性吸收。用抽样次数反推试过的 n 序列(每轮 n 次抽样)。"""
    calls = []

    def never_win():
        calls.append(1)
        return 1.0                      # 恒不落入胜/负区间 -> 无不一致对 -> 功效0

    assert ce.required_tasks(0.5, 0.1, sims=1, n_max=64, rand=never_win) is None
    expected = [8, 12, 16, 22, 30, 42, 58]      # 8 -> max(12, 11) -> max(16,16) -> ...
    assert len(calls) == sum(expected), \
        f"抽样总数应为 {sum(expected)}(即试过 {expected}), 实得 {len(calls)}"


def test_mde_search_upper_bound_inclusive():
    """搜索上界是闭区间: p_win + p_loss 恰好等于 1.0 时仍须尝试。
    构造: step=0.5 时网格为 0.5 -> 1.0, 恒 0.5 的 rand 使 0.5 全平局、1.0 全胜。
    上界改成开区间后 1.0 不会被尝试, 结果由 1.0 变成 None。"""
    mde = ce.detectable_effect(20, p_loss=0.0, step=0.5, sims=3, rand=lambda: 0.5)
    assert mde == 1.0, f"p_win=1.0 必须被尝试并达标: {mde!r}"


def test_mde_counter_init_exact():
    """模拟计数器必须从 0 起: n=6 全胜时不一致对为 (6,0), p=0.031 显著;
    初值若为 1 则是 (7,1), p=0.070 不显著 -> MDE 由 0.1 变成 None。
    这是把"统计上大概对"变成精确断言的例子。"""
    assert abs(ce.mcnemar_exact(6, 0) - 0.03125) < 1e-9
    assert abs(ce.mcnemar_exact(7, 1) - 0.0703125) < 1e-9
    mde = ce.detectable_effect(6, step=0.1, sims=3, rand=lambda: 0.0)
    assert mde == 0.1, f"(6,0) 显著 -> 首个网格点即达标: {mde!r}"




def test_planner_validation_boundaries():
    """参数校验的每个边界都要钉住(此前只测过 p_loss>p_win 一种):
    相等、超上界、和超过1 都必须拒绝; 和恰好等于1 必须放行。"""
    for p_w, p_l in [(0.5, 0.5), (0.4, 0.5), (1.5, 0.1), (0.6, 0.5), (-0.1, 0.0)]:
        try:
            ce.required_tasks(p_w, p_l, sims=2, n_max=8)
            assert False, f"非法参数应拒绝: p_win={p_w}, p_loss={p_l}"
        except ValueError:
            pass
    # p_win + p_loss 恰好等于 1: 合法(闭区间), 恒胜时最小网格点即达标
    assert ce.required_tasks(0.6, 0.4, sims=2, n_max=8, rand=lambda: 0.0) == 8
    # MDE 侧的校验边界
    for n, p_l in [(0, 0.0), (1, 1.0), (1, -0.1)]:
        try:
            ce.detectable_effect(n, p_loss=p_l, sims=2)
            assert False, f"非法参数应拒绝: n={n}, p_loss={p_l}"
        except ValueError:
            pass


def test_simulation_comparison_boundaries():
    """模拟内部两处比较都是开区间: 抽样值恰好等于 p_win(或 p_win+p_loss)时算平局,
    不算胜(负)。改成闭区间后, 恒定抽样会从"全平局"翻转成"全胜(全负)", 结论完全不同。"""
    # r 恰等于 p_win: 原版全平局 -> 无不一致对 -> 不可达
    assert ce.required_tasks(0.5, 0.0, sims=3, n_max=16, rand=lambda: 0.5) is None
    # r 恰等于 p_win+p_loss: 同样全平局(改成 <= 会变成全负 -> 显著)
    assert ce.required_tasks(0.3, 0.2, sims=3, n_max=16, rand=lambda: 0.5) is None
    # 略小于 p_win 才算胜: 确认构造本身有区分力
    assert ce.required_tasks(0.5, 0.0, sims=3, n_max=16, rand=lambda: 0.49) == 8




def test_expr_whitelist_not_redundant():
    """白名单正则不是可有可无的第二道锁: 空 builtins 沙箱只挡住 __import__ 这类,
    而"1 and 2"这种纯语法表达式沙箱照样执行。此前的恶意探针恰好被沙箱挡住,
    于是把正则检查短路掉(or 改 and)后存活 —— 正则的作用一直没被验证。"""
    obs = [{"tool_call_id": "tc_1", "observation": "x"}]
    sneaky = lambda p, s, sch: {"computable": True, "expression": "1 and 2",
                               "claimed_value": 2, "inputs": []}
    r = ce.verify_derived("绕过沙箱的合法表达式", obs, sneaky)
    assert r["verdict"] == "bad-expression", \
        f"非算术表达式必须被正则拦下(而不是靠沙箱): {r['verdict']}"
    boolish = lambda p, s, sch: {"computable": True, "expression": "True + 1",
                                 "claimed_value": 2, "inputs": []}
    assert ce.verify_derived("布尔混入", obs, boolish)["verdict"] == "bad-expression"


def test_pass_hat_k_equality_and_smoothing():
    """两处从未被覆盖的精确值: successes 恰好等于 k; 置换检验的 +1 平滑。"""
    # successes == k: 边界既非"不足"也非"全过", 必须给 C(k,k)/C(n,k)
    assert abs(ce.pass_hat_k(4, 8, 4) - 1 / 70) < 1e-12
    assert abs(ce.pass_hat_k(3, 5, 3) - 1 / 10) < 1e-12
    # +1 平滑保证 p 永不为 0(否则会报出"绝对确定"的假结论)
    c = ce.paired_compare([1.0] * 20, [0.0] * 20, n_resamples=100)
    assert abs(c["p_value"] - 1 / 101) < 1e-12, \
        f"最极端情形的 p 应为 1/(n+1)=1/101, 得到 {c['p_value']}"
    assert c["p_value"] > 0, "p 值永不为零"


def test_planner_extreme_valid_inputs():
    """两个合法极值此前从未被接受性测试: p_win=1.0(必胜, 需 p_loss=0);
    detectable_effect(n=1)(合法但功效不可达, 应返回 None 而不是报错)。"""
    assert ce.required_tasks(1.0, 0.0, sims=2, n_max=8, rand=lambda: 0.0) == 8
    assert ce.detectable_effect(1, sims=2, rand=lambda: 0.0) is None, \
        "n=1 是合法输入: 最多1个不一致对, 永不显著 -> None"
    # MDE 的 hits 计数器必须从 0 起: sims=1 时初值1会让功效直接变成 1.0
    assert ce.detectable_effect(20, sims=1, rand=lambda: 1.0) is None, \
        "恒不胜 -> 功效 0 -> 不可达; hits 初值若为 1 则会误判首个网格点达标"




def test_interpret_forces_null_bounds():
    """null 结论必须自带"能排除多大效应" —— 本仓库曾把 16 单元(MDE=0.46)的
    Δ=0.000 写成"效应根本不存在"。三个分支: 显著 / 有界的null / 无信息的null。"""
    sig = ce.interpret(ce.paired_compare([1.0] * 20, [0.0] * 20))
    assert sig["verdict"] == "significant" and sig["rules_out"] is None
    assert "显著" in sig["text"] and "CI95" in sig["text"]
    # 小样本 null: 两对都有非零差值时地板是 0.5 > alpha, 故诊断为"检验无力"
    # —— 先从"无信息"改到这里(docs/lessons.md#p-floor), 后补上"必须真有非零差值"这个前提(docs/lessons.md#floor-basis)。
    weak = ce.interpret(ce.paired_compare([1.0, 0.0], [0.0, 1.0]))
    assert weak["verdict"] == "null" and weak["rules_out"] is None
    assert "检验无力" in weak["text"] and weak["p_floor"] == 0.5
    # 两侧完全相同 = 零有效对: 地板是 1.0, 病因是"没有任何非零差值"而非"样本少"
    same = ce.interpret(ce.paired_compare([1.0, 0.0], [1.0, 0.0]))
    assert same["p_floor"] == 1.0 and "非零差值对只有 1 个" in same["text"], same["text"]
    # 真正的"无信息": 连单元都没有(合成报告), 地板无从谈起
    none_units = ce.interpret(ce.paired_compare([1.0], [1.0]), n_units=0)
    assert "无信息" in none_units["text"] and none_units["p_floor"] is None
    # 大样本 null: 必须给出具体可排除范围。显式 floor_n 表明这批 80 单元全都有效
    tight = ce.interpret(ce.paired_compare([1.0, 0.0] * 8, [1.0, 0.0] * 8),
                         n_units=80, floor_n=80)
    assert tight["verdict"] == "null" and tight["rules_out"] == 0.1
    assert "只能排除" in tight["text"] and "10%" in tight["text"]
    assert tight["n_units"] == 80, "n_units 可显式指定(逐轮单元数≠逐题数)"
    # 默认取 compare["n"]
    d = ce.interpret(ce.paired_compare([1.0] * 5, [0.0] * 5))
    assert d["n_units"] == 5
    # alpha 可调: 放宽后同一结果可能翻为显著
    borderline = ce.paired_compare([1.0] * 6, [0.0] * 6)
    assert ce.interpret(borderline, alpha=0.001)["verdict"] == "null"
    assert ce.interpret(borderline, alpha=0.05)["verdict"] == "significant"




def test_interpret_boundaries():
    """三处边界: p 恰好等于 alpha 算不显著(严格小于); n_units=1 仍要给出结论;
    n_units=0 不得去跑 MDE(会崩)。棘轮在 interpret 入库时抓到这三处缺测。"""
    c = ce.paired_compare([1.0] * 20, [0.0] * 20)
    p = c["p_value"]                       # 1/10001
    # alpha 恰好等于 p: 严格小于 -> 判 null(不显著)
    at = ce.interpret(c, alpha=p)
    assert at["verdict"] == "null", f"p == alpha 应判不显著: p={p}"
    just_above = ce.interpret(c, alpha=p * 1.0001)
    assert just_above["verdict"] == "significant", "alpha 略大于 p 才算显著"
    # n_units=1: 合法, 走 MDE 分支(1 个单元必然不可达)
    one = ce.interpret(c, n_units=1, alpha=p)
    assert one["verdict"] == "null" and one["rules_out"] is None and one["n_units"] == 1
    # n_units=0: 不得调用 detectable_effect(它要求 n>=1), 应直接给无信息结论
    zero = ce.interpret(c, n_units=0, alpha=p)
    assert zero["verdict"] == "null" and zero["rules_out"] is None and zero["n_units"] == 0
    assert "无信息" in zero["text"]




def test_p_floor_and_min_units():
    """p 地板: n 个配对单元下再大的效应也拿不到比 2/2^n 更小的 p。
    实测踩坑(docs/lessons.md#p-floor): 3 案例得 Δ=+0.911 CI 排除 0 十万八千里, p 却是 0.252。"""
    assert ce.p_floor(1) == 1.0, "1 个单元: 两种符号排列, 双侧 p 恒为 1"
    assert ce.p_floor(3) == 0.25 and ce.p_floor(6) == 2 / 64
    assert ce.p_floor(7) < 0.02
    # 大 n 时重采样下限接管, 不再是 2/2^n(那会下溢成 0)
    assert ce.p_floor(100, resamples=999) == 1 / 1000
    assert ce.p_floor(200) == ce.p_floor(100) > 0, "极大 n 不得返回 0"
    # 与实际检验一致: 完美分离时实测 p 应贴着地板(重采样有 ±1 计数抖动)
    for n in (3, 4, 5, 6):
        got = ce.paired_compare([1.0] * n, [0.0] * n)["p_value"]
        assert got >= ce.p_floor(n) * 0.95, f"n={n}: 实测 p={got} 低于地板 {ce.p_floor(n)}"
        assert got <= ce.p_floor(n) * 1.5, f"n={n}: 完美分离的 p 应贴着地板, 得 {got}"
    try:
        ce.p_floor(0)
        raise AssertionError("n=0 应报错")
    except ValueError as e:
        assert "必须 >=1" in str(e)
    # 最小单元数: alpha 越严要求越多
    assert ce.min_units_for_alpha(0.05) == 6 and ce.min_units_for_alpha(0.01) == 8
    assert ce.min_units_for_alpha(0.5) < ce.min_units_for_alpha(0.05)
    for bad in (0, 1, -0.1, 1.5):
        try:
            ce.min_units_for_alpha(bad)
            raise AssertionError(f"alpha={bad} 应报错")
        except ValueError:
            pass


def test_interpret_distinguishes_powerless_test_from_null():
    """三种"没显著"必须分开: 检验无力(p 地板 > alpha) / 无信息(MDE 不可达) /
    有界的 null。混为一谈会让"样本不够"冒充"没有差异"。"""
    # 效应巨大但 n=3: 必须说"检验无力", 且给出所需单元数, 不许报 rules_out
    weak = ce.interpret(ce.paired_compare([1.0] * 3, [0.0] * 3))
    assert weak["verdict"] == "null" and weak["rules_out"] is None
    assert "检验无力" in weak["text"] and "需至少 6 个配对单元" in weak["text"]
    assert weak["p_floor"] == 0.25
    assert "+1.000" in weak["text"], "无力时仍须报点估计与 CI, 否则读者以为没效应"
    # n=6 同样效应: 地板降到 0.031 < 0.05, 于是判显著
    ok = ce.interpret(ce.paired_compare([1.0] * 6, [0.0] * 6))
    assert ok["verdict"] == "significant"
    # n=6 且六对都有非零差值、但方向混杂 -> 走有界 null 分支(有 rules_out, 非"无力")
    # 注意不能用两侧完全相同的数据: 那是零有效对, 会正确判为"检验无力"(docs/lessons.md#floor-basis)。
    tie = ce.interpret(ce.paired_compare([0.6, 0.4] * 3, [0.4, 0.6] * 3))
    assert tie["verdict"] == "null" and "检验无力" not in tie["text"], tie["text"]
    assert tie["p_floor"] == 2 / 64 and tie["rules_out"] is not None
    # alpha 更严时同一 n=6 会翻回"检验无力"
    strict = ce.interpret(ce.paired_compare([1.0] * 6, [0.0] * 6), alpha=0.01)
    assert "检验无力" in strict["text"] and "需至少 8 个配对单元" in strict["text"]




def test_p_floor_knife_edges():
    """地板恰好等于 alpha 时显著性仍不可达(判据是 p < alpha) —— 变异测试
    抓到这处: 原本写的 floor > alpha 会把这种设计放行为"可能显著"。"""
    a = 2 / 64                                   # 恰好是 p_floor(6)
    assert ce.p_floor(6) == a
    # n=6 且 alpha 恰好等于地板: 必须判"检验无力", 并要求更多单元
    r = ce.interpret(ce.paired_compare([1.0] * 6, [0.0] * 6), alpha=a)
    assert "检验无力" in r["text"], r["text"]
    assert "需至少 7 个配对单元" in r["text"], r["text"]
    # alpha 略大于地板并不够: 地板 2/2^n 是理论下界, 重采样实测 p 落在其上方
    # (n=6 完美分离: 地板 0.03125, 实测 ~0.032)。故 alpha=地板*1.01 仍不显著。
    c6 = ce.paired_compare([1.0] * 6, [0.0] * 6)
    assert a < c6["p_value"] < a * 1.2, f"实测 p 应略高于地板: {c6['p_value']} vs {a}"
    assert ce.interpret(c6, alpha=a * 1.01)["verdict"] == "null"
    assert ce.interpret(c6, alpha=a * 1.5)["verdict"] == "significant"
    # min_units 与地板一致(严格小于), 它算的是理论下界, 不含重采样抖动
    assert ce.min_units_for_alpha(a) == 7 and ce.min_units_for_alpha(a * 1.01) == 6
    # n_units=1 且显式 floor_n=1: 诊断是"检验无力"(地板 1.0), 不是"无信息"
    # —— 后者专指没有单元。不给 floor_n 时会用 compare 自报的有效对数(此例 3), 更准。
    one = ce.interpret(ce.paired_compare([1.0] * 3, [0.0] * 3), n_units=1, floor_n=1)
    assert "检验无力" in one["text"] and one["p_floor"] == 1.0
    assert "无信息" not in one["text"]
    auto = ce.interpret(ce.paired_compare([1.0] * 3, [0.0] * 3), n_units=1)
    assert auto["p_floor"] == ce.p_floor(3), "未给 floor_n 时应采用有效对数"
    # 逐步递增不得跳号: 每个 alpha 的答案都要精确
    assert [ce.min_units_for_alpha(x) for x in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)] == \
        [3, 4, 5, 6, 7, 8]
    # alpha 边界的报错措辞要点明范围, 否则与"低于重采样下限"混淆
    for bad in (0, 1):
        try:
            ce.min_units_for_alpha(bad)
            raise AssertionError(f"alpha={bad} 应报错")
        except ValueError as e:
            assert "必须在 (0,1)" in str(e), str(e)
    try:
        ce.min_units_for_alpha(1e-9, resamples=100)
        raise AssertionError("低于重采样下限时应报错")
    except ValueError as e:
        assert "需增大 resamples" in str(e), str(e)




def test_p_floor_resample_region_is_exact():
    """中段 n(排列地板已低于重采样地板, 但 n 仍 <64)必须取重采样地板的精确值。
    变异测试抓到这段没被覆盖: 把 1/(resamples+1) 写成 2/(resamples+1) 或
    1/(resamples+2) 都能通过原有断言 —— 因为原测试只探了 n<15 与 n>=64 两头。"""
    # 交叉点: 2/2^n 何时降到重采样地板以下
    assert ce.p_floor(14, 10000) == 2 / 2 ** 14, "n=14 时排列地板仍占优"
    assert ce.p_floor(20, 10000) == 1 / 10001, "n=20 时重采样地板接管, 且须精确"
    assert ce.p_floor(40, 10000) == 1 / 10001
    assert ce.p_floor(63, 10000) == 1 / 10001, "63 仍走 max 分支"
    assert ce.p_floor(64, 10000) == 1 / 10001, "64 走 else 分支, 值必须相同"
    # 换 resamples: 分子分母都要对
    assert ce.p_floor(20, 999) == 1 / 1000 and ce.p_floor(11, 999) == 1 / 1000
    assert ce.p_floor(10, 999) == 2 / 2 ** 10, "n=10 时 2/1024 仍低于 1/1000"
    # 单调不增: 地板不会因 n 变大而升高
    vals = [ce.p_floor(n) for n in range(1, 70)]
    assert all(b <= a for a, b in zip(vals, vals[1:])), "p_floor 必须单调不增"


def test_min_units_error_reports_actual_floor():
    """报错必须给出真实的重采样地板值 —— 读者据此决定把 resamples 提到多少。
    写错这个数(2/(r+1) 或 1/(r+2))不影响任何行为, 只误导人, 故必须断言。"""
    for rs in (100, 999, 10000):
        try:
            ce.min_units_for_alpha(1e-12, resamples=rs)
            raise AssertionError("应报错")
        except ValueError as e:
            assert repr(1 / (rs + 1)) in str(e) or f"{1 / (rs + 1)}" in str(e), \
                f"resamples={rs}: 报错未含真实地板 {1 / (rs + 1)}: {e}"




def test_paired_compare_reports_effective_pairs():
    """置换检验的可达最小 p 由非零差值对数决定 —— 符号翻转对零差值对是恒等操作。
    实证(docs/lessons.md#floor-basis): 5 对零差 + 1 对满差, p=1.0000 恰是 p_floor(1), 与 p_floor(6) 无关。
    这与 McNemar 那处是同一个 bug 类: 接口上传了语义不同的同名量。"""
    c = ce.paired_compare([1.0] * 6, [1.0] * 5 + [0.0])
    assert c["n"] == 6 and c["n_effective"] == 1 and c["ties"] == 5
    assert c["p_value"] == 1.0, "只有一对非零时置换检验必然给 1.0"
    assert c["p_value"] == ce.p_floor(c["n_effective"]), "实测 p 应恰等于有效对的地板"
    # 全非零时两个基数相等
    c2 = ce.paired_compare([0.5] * 6, [0.4] * 6)
    assert c2["n_effective"] == c2["n"] == 6 and c2["ties"] == 0
    # 有效对数的定义: 与 wins+losses 恒等(平局即零差)
    for a, b in (([1.0, 0.0, 1.0], [0.0, 0.0, 1.0]), ([0.3] * 4, [0.3, 0.1, 0.3, 0.9])):
        c3 = ce.paired_compare(a, b)
        assert c3["n_effective"] == c3["wins"] + c3["losses"], (a, b, c3)
        assert c3["n_effective"] + c3["ties"] == c3["n"]


def test_interpret_basis_priority_and_wording():
    """地板基数三选一: 显式 floor_n > compare 的 n_effective > 单元数。
    三者的补救处方不同, 措辞必须点明是哪一种, 否则读者会加错东西。"""
    # 1 有零差值对 -> 用有效对数, 措辞"非零差值对"
    c = ce.paired_compare([1.0] * 18, [1.0] * 15 + [0.6, 0.83, 0.5])
    v = ce.interpret(c)
    assert v["verdict"] == "null" and v["p_floor"] == ce.p_floor(3)
    assert "非零差值对只有 3 个" in v["text"], v["text"]
    assert "配对单元" not in v["text"] and "不一致对" not in v["text"]
    # 2 显式 floor_n 优先(report 的 McNemar 路径), 措辞"不一致对"
    v2 = ce.interpret(c, floor_n=2)
    assert v2["p_floor"] == ce.p_floor(2) and "不一致对只有 2 个" in v2["text"]
    # 3 无 n_effective 字段(合成 compare) -> 退回单元数, 措辞"配对单元"
    synth = {"n": 3, "mean_diff": 0.5, "diff_ci": (0.4, 0.6), "p_value": 0.3}
    v3 = ce.interpret(synth)
    assert v3["p_floor"] == ce.p_floor(3) and "配对单元只有 3 个" in v3["text"]
    # 4 全非零时不该用"非零差值对"措辞(它等于单元数, 说成配对单元才不误导)
    c4 = ce.paired_compare([0.5, 0.5], [0.4, 0.4])
    v4 = ce.interpret(c4)
    assert "配对单元只有 2 个" in v4["text"], v4["text"]
    # 5 有效对数为 0(两侧完全相同) -> 夹到 1, 地板 1.0, 不得崩
    c5 = ce.paired_compare([1.0] * 4, [1.0] * 4)
    v5 = ce.interpret(c5)
    assert c5["n_effective"] == 0 and v5["p_floor"] == 1.0
    assert v5["verdict"] == "null" and "检验无力" in v5["text"]




def test_required_pairs_contracts():
    """连续分数样本量规划的契约。用注入的确定性 gauss 钉死边界, 不靠"统计上大概对"
    —— 与 required_tasks/detectable_effect 同一套做法(docs/lessons.md#injected-rand)。"""
    # 差值恒定(sd=0): 每对同号, 置换 p 恒等于地板, 故样本量只由地板决定
    d = ce.required_pairs(0.1, 0.0, detail=True)
    assert d["n"] == d["floor_n"] == ce.min_units_for_alpha(0.05, 2000)
    assert d["achieved_power"] == 1.0
    assert ce.required_pairs(0.1, 0.0) == d["n"], "非 detail 应返回同一个 n"
    # 起点不得低于 p 地板 —— 否则规划出的设计再大效应也拿不到显著
    always = ce.required_pairs(1.0, 0.001, sims=3, resamples=400, gauss=lambda mu, s: mu, detail=True)
    assert always["n"] >= always["floor_n"] >= 6, always
    # alpha 更严 -> 地板更高 -> 起点更高
    strict = ce.required_pairs(1.0, 0.001, alpha=0.01, sims=3, resamples=400,
                               gauss=lambda mu, s: mu, detail=True)
    assert strict["floor_n"] > always["floor_n"], (strict, always)
    # 效应过小时返回 None(而非无限搜索), detail 版仍带 floor_n 供诊断
    zero = lambda mu, s: 0.0        # 差值恒为 0 -> 永远检不出
    none_d = ce.required_pairs(0.001, 1.0, n_max=16, sims=4, resamples=200,
                               gauss=zero, detail=True)
    assert none_d["n"] is None and none_d["achieved_power"] is None
    assert none_d["floor_n"] == 6, "失败时也要交代地板, 否则调用者不知从何起算"
    assert ce.required_pairs(0.001, 1.0, n_max=16, sims=4, resamples=200,
                             gauss=zero) is None
    # 输入契约
    for bad_md, bad_sd in ((0, 0.2), (0.0, 1.0)):
        try:
            ce.required_pairs(bad_md, bad_sd)
            raise AssertionError("mean_diff=0 应报错")
        except ValueError as e:
            assert "无效应" in str(e), str(e)
    try:
        ce.required_pairs(0.1, -0.1)
        raise AssertionError("sd<0 应报错")
    except ValueError as e:
        assert "sd 不能为负" in str(e), str(e)
    # 单调性: 效应越小需要越多对(同 sd)。用注入的确定性采样器 —— 真跑蒙特卡洛要 15 秒,
    # 会把 pre-commit 依赖的 2 秒快速套件拖垮(docs/lessons.md#fast-suite-budget)。慢的那份留给下一个测试。
    det = lambda mu, s: mu          # 差值恒为 mu: 功效由地板与效应符号决定
    big = ce.required_pairs(0.5, 0.3, sims=6, resamples=400, gauss=det)
    small = ce.required_pairs(0.05, 0.3, sims=6, resamples=400, gauss=det)
    assert big is not None and small is not None and big <= small, (big, small)


def calibrate_required_pairs_against_formula():
    """规划器必须与教科书功效公式同量级 —— 偏离一倍以上说明实现有错。

    刻意不叫 test_*: 真跑蒙特卡洛要 4~5 秒, 会把 pre-commit 依赖的 2 秒快速套件拖慢
    一倍以上。由 checkall.sh 的慢层调用(与 check_hooks.py 同层理由)。
    教训正藏在这里(docs/lessons.md#power-formula): 手算 (1.96*sd/Δ)^2 答的是"CI 恰好排除 0"(实测功效仅
    0.45), 而功效公式是 ((z_a/2 + z_power)*sd/Δ)^2, 两者差 2.04 倍。"""
    for md, sd in ((0.122, 0.264), (0.3, 0.3)):
        formula = ((1.96 + 0.8416) * sd / md) ** 2
        got = ce.required_pairs(md, sd, sims=80, resamples=1000)
        assert got is not None
        assert 0.7 * formula <= got <= 2.0 * formula, \
            f"Δ={md} sd={sd}: 公式 {formula:.0f} vs 规划器 {got} —— 偏离过大"
        # 关键: 必须明显大于"CI 排除 0"的那个数(它只有约一半功效)
        assert got > (1.96 * sd / md) ** 2, "规划器不得退化成 CI 半宽公式"




def test_required_pairs_grid_and_counting_boundaries():
    """变异测试在 required_pairs 上抓到 8 个缺口, 7 个可精确钉死(第 8 个是 p 恰等于 alpha,
    属已知等价类"实践中不可达")。全部用注入的确定性采样器 —— 采样器按调用序号分段,
    就能精确控制"哪几次 sim 通过", 从而把功效算到小数点后一位。"""
    R = 200                                            # 重采样数: 地板 min_units=6
    # 1 达标比较是 >=: 恰好 80% 的 sim 通过时必须在该 n 返回, 且 achieved_power 精确为 0.8
    calls = {"i": 0}

    def four_of_five(mu, s):
        k = calls["i"] // 6                            # n=6 时每 6 次调用是一个 sim
        calls["i"] += 1
        return mu if k % 5 != 4 else 0.0               # 第 5 个 sim 差值全零 -> p=1 -> 不过

    d = ce.required_pairs(1.0, 0.1, sims=5, resamples=R, gauss=four_of_five, detail=True)
    assert d["n"] == 6 and d["achieved_power"] == 0.8, d
    # 2 计数: 全过时 achieved_power 必须恰为 1.0(初值 0、每次 +1 —— 改成 1 或 +2 都会超过 1)
    d2 = ce.required_pairs(1.0, 0.1, sims=3, resamples=R, gauss=lambda mu, s: mu, detail=True)
    assert d2["achieved_power"] == 1.0 and d2["n"] == 6, d2
    # 3 计数初值: 永不通过且只有 1 个 sim 时, 功效必须是 0 而非 1/1 —— 否则返回 6 而非 None
    assert ce.required_pairs(1.0, 0.1, sims=1, resamples=R, n_max=16,
                             gauss=lambda mu, s: 0.0) is None
    # 4 搜索上界是 <=: n_max 恰等于地板时仍要评估该点, 不得直接返回 None
    assert ce.required_pairs(1.0, 0.1, sims=2, resamples=R, n_max=6,
                             gauss=lambda mu, s: mu) == 6
    # 5 网格步长: 地板 6 之后的下一个点是 max(6+2, int(6*1.15)) = 8。
    #   让前 sims*6 次调用全零(n=6 不过), 之后全通过 -> 返回值就是第二个网格点。
    cnt = {"i": 0}

    def fail_first_grid(mu, s):
        cnt["i"] += 1
        return 0.0 if cnt["i"] <= 3 * 6 else mu

    assert ce.required_pairs(1.0, 0.1, sims=3, resamples=R, gauss=fail_first_grid) == 8, \
        "第二个网格点应为 8(步长 +2 与 x1.15 取大); 步长改动会让超调加大"
    # 6 未注入采样器时用 seed 驱动的正态: 不得崩, 且同 seed 结果可复现
    a = ce.required_pairs(3.0, 0.01, sims=2, resamples=R, n_max=8)
    b = ce.required_pairs(3.0, 0.01, sims=2, resamples=R, n_max=8)
    assert a == b == 6, (a, b)


def test_fmt_p_keeps_magnitude_of_tiny_p():
    """极小 p 不得打成 "0.0000"(读者会当成零): 边界 0.0001 仍用四位小数, 之下切到科学计数。
    demo 里 McNemar p=3e-05 曾被显示为 p=0.0000。"""
    assert ce.fmt_p(0.5) == "0.5000" and ce.fmt_p(1.0) == "1.0000"
    assert ce.fmt_p(0.0001) == "0.0001", "恰好等于边界仍用小数"
    assert ce.fmt_p(0.00009999) == "1.0e-04" and ce.fmt_p(3e-05) == "3.0e-05"
    assert "0.0000" not in ce.fmt_p(1 / 10001)
    # interpret 的显著分支用它: 完美分离 20 对的 p=1/10001 应显示为 1.0e-04 而非 0.0001
    text = ce.interpret(ce.paired_compare([1.0] * 20, [0.0] * 20))["text"]
    assert "p=1.0e-04" in text, text


def test_interpret_english_covers_all_verdicts_without_cjk():
    """英文报告必须覆盖全部四种结论且不含任何中文字符 —— 漏译一处就会在英文 README 里露出中文。
    两种语言的占位符集合也必须一致, 否则某个数据只在一种语言里被报出来。"""
    import re
    import string
    cjk = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
    cases = {
        "significant": ce.interpret(ce.paired_compare([1.0] * 20, [0.0] * 20), lang="en"),
        "powerless_units": ce.interpret(ce.paired_compare([1.0] * 3, [0.0] * 3), lang="en"),
        "powerless": ce.interpret(ce.paired_compare([1.0] * 18, [1.0] * 15 + [0.6, 0.83, 0.5]), lang="en"),
        "bounded": ce.interpret(ce.paired_compare([1.0, 0.0] * 8, [1.0, 0.0] * 8), n_units=80, floor_n=80, lang="en"),
        "uninformative": ce.interpret(ce.paired_compare([1.0], [1.0]), n_units=0, lang="en"),
        "powerless_discordant": ce.interpret(ce.paired_compare([1.0] * 3, [0.0] * 3), floor_n=2, lang="en"),
    }
    expect = {"significant": "significant: Δ=", "powerless": "powerless test: only 3 nonzero-difference pairs",
              "powerless_units": "powerless test: only 3 paired units",
              "bounded": "no difference detected", "uninformative": "uninformative null",
              "powerless_discordant": "only 2 discordant pairs"}
    for k, v in cases.items():
        assert expect[k] in v["text"], (k, v["text"])
        assert not cjk.search(v["text"]), f"{k}: 英文文案含中文: {v['text']}"
    assert "needs at least 6 nonzero-difference pairs" in cases["powerless"]["text"]
    # verdict 字段与语言无关
    zh = ce.interpret(ce.paired_compare([1.0] * 18, [1.0] * 15 + [0.6, 0.83, 0.5]))
    assert zh["verdict"] == cases["powerless"]["verdict"] == "null" and zh["p_floor"] == cases["powerless"]["p_floor"]
    # 占位符集合一致
    fields = lambda s: {f for _, f, _, _ in string.Formatter().parse(s) if f}
    for key in ce._MSG["zh"]:
        assert fields(ce._MSG["zh"][key]) == fields(ce._MSG["en"][key]), f"占位符不一致: {key}"
    assert set(ce._MSG["zh"]) == set(ce._MSG["en"])


def test_set_language_switches_default_and_validates():
    prev = ce.DEFAULT_LANG
    try:
        assert ce.set_language("en") == "zh"
        assert ce.DEFAULT_LANG == "en"
        assert "significant:" in ce.interpret(ce.paired_compare([1.0] * 20, [0.0] * 20))["text"]
        assert "显著" in ce.interpret(ce.paired_compare([1.0] * 20, [0.0] * 20), lang="zh")["text"], "显式 lang 优先"
        try:
            ce.set_language("fr")
            raise AssertionError("不支持的语言应报错")
        except ValueError as e:
            assert "fr" in str(e) and "en" in str(e)
        assert ce.DEFAULT_LANG == "en", "报错不得改变默认值"
    finally:
        ce.set_language(prev)
    assert ce.DEFAULT_LANG == prev


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
