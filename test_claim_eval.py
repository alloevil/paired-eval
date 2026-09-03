# -*- coding: utf-8 -*-
"""claim_eval 离线结构测试:mock llm/search,零网络零API,验证纯代码路径。
运行: python3 test_claim_eval.py
"""
import claim_eval as ce

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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
