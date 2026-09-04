# -*- coding: utf-8 -*-
"""eval_task 离线测试:mock llm/search,零网络。运行: python3 test_eval_task.py"""
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 项目根: 让 `python3 tests/x.py` 直接可跑
from paired_eval import claim_eval as ce
from paired_eval import eval_task as et
from paired_eval import rubric_eval as re_


def mock_search(query):
    return f"evidence({query})"


def mock_llm(prompt, system, schema):
    if system is ce.EXTRACT_SYSTEM:
        return {"claims": [
            {"text": "K真事实", "source_quote": "K", "verifiable": True,
             "importance": "core", "search_query": "qK"},
            {"text": "W假陈述", "source_quote": "W", "verifiable": True,
             "importance": "detail", "search_query": "qW"}]}
    if system is ce.JUDGE_WORLD_SYSTEM:
        v = "contradicted" if "W假陈述" in prompt else "supported"
        return {"verdict": v, "evidence_quote": "q", "reasoning": "r"}
    if system is ce.JUDGE_TRAJ_SYSTEM:
        v = "grounded" if "K真事实" in prompt else "fabricated"
        return {"verdict": v, "source_tool_call": "tc_1" if v == "grounded" else None,
                "evidence_quote": "q", "reasoning": "r"}
    raise AssertionError("未知system")


T_EXACT = {"id": "t-exact", "verification":
           {"class": "exact", "gold": "8635亿", "kind": "numeric"}}
T_RETR = {"id": "t-retr", "verification": {"class": "retrieval"}}
T_TRAJ = {"id": "t-traj", "verification": {"class": "trajectory"}}
OBS = [{"tool_call_id": "tc_1", "tool": "s", "observation": "o"}]


def test_exact_route():
    r = et.evaluate(T_EXACT, response="推理...答案: 863,500,000,000元")
    assert r["score"] == 1.0 and r["verdict"] == "correct"
    r2 = et.evaluate(T_EXACT, response="答案: 12亿")
    assert r2["score"] == 0.0 and r2["verdict"] == "incorrect"
    r3 = et.evaluate(T_EXACT, response="不知道")
    assert r3["score"] == 0.0 and r3["verdict"] == "not_attempted", "弃答语义保留"


def test_retrieval_route():
    r = et.evaluate(T_RETR, response="DOC", llm=mock_llm, search=mock_search)
    # supported=K(core,3), contradicted=W(detail,1) -> weighted_precision=3/4
    assert abs(r["score"] - 0.75) < 1e-9
    assert r["metrics"]["supported"] == 1 and r["metrics"]["contradicted"] == 1


def test_trajectory_route():
    r = et.evaluate(T_TRAJ, response="DOC", llm=mock_llm, observations=OBS)
    assert abs(r["score"] - 0.5) < 1e-9, "K grounded, W fabricated -> grounding_rate=0.5"
    assert r["metrics"]["fabrication_rate"] == 0.5


def test_validation():
    for bad, msg in [
        ({"id": "x"}, "缺 verification"),
        ({"id": "x", "verification": {"class": "preference"}}, "不支持的类别"),
        ({"id": "x", "verification": {"class": "exact"}}, "exact 缺 gold"),
        ({"id": "x", "verification": {"class": "rubric", "criteria": []}}, "rubric 空criteria"),
    ]:
        try:
            et.validate_task(bad)
            assert False, msg
        except ValueError:
            pass
    try:
        et.evaluate(T_RETR, response="DOC")  # 缺 llm/search
        assert False, "缺依赖必须报错而非静默"
    except ValueError:
        pass


def test_batch_feeds_paired_compare():
    items = [{"task": T_EXACT, "response": "答案: 8635亿"},
             {"task": T_EXACT, "response": "答案: 1亿"}]
    rows = et.evaluate_batch(items)
    scores = [r["score"] for r in rows]
    assert scores == [1.0, 0.0]
    # score 序列可直接进配对比较
    cmp = ce.paired_compare(scores, [0.0, 0.0])
    assert cmp["wins"] == 1 and cmp["n"] == 2


def test_rubric_route():
    from paired_eval import rubric_eval as re_

    def rubric_llm(prompt, system, schema):
        assert system is re_.JUDGE_RUBRIC_SYSTEM
        v = "met" if "条A" in prompt else "not_met"
        return {"verdict": v, "reasoning": "r"}

    t = {"id": "t-rub", "verification": {"class": "rubric", "criteria": [
        {"text": "条A", "weight": 3}, {"text": "条B", "weight": 1}]}}
    r = et.evaluate(t, response="RESP", llm=rubric_llm)
    assert abs(r["score"] - 0.75) < 1e-9 and r["metrics"]["met"] == 1
    try:
        et.evaluate(t, response="RESP")  # 缺 llm
        assert False, "rubric 缺 llm 必须报错"
    except ValueError:
        pass


def test_verifier_fingerprint():
    fp1 = et.verifier_fingerprint()
    assert fp1 == et.verifier_fingerprint() and len(fp1) == 12, "同版本指纹稳定"
    r = et.evaluate(T_EXACT, response="答案: 8635亿")
    assert r["verifier_fp"] == fp1, "每条结果都携带验证器指纹"
    old = ce.JUDGE_WORLD_SYSTEM
    try:
        ce.JUDGE_WORLD_SYSTEM = old + "改动"
        assert et.verifier_fingerprint() != fp1, "prompt 改动必须换指纹"
    finally:
        ce.JUDGE_WORLD_SYSTEM = old


def test_meter_attribution():
    m = ce.Meter()
    r1 = et.evaluate(T_EXACT, response="答案: 8635亿", meter=m)
    assert r1["cost"] == {"llm_calls": 0, "llm_chars_in": 0,
                          "llm_chars_out": 0, "search_calls": 0}, "exact 路由零LLM成本"
    r2 = et.evaluate(T_RETR, response="DOC", llm=mock_llm, search=mock_search, meter=m)
    assert r2["cost"]["llm_calls"] == 3 and r2["cost"]["search_calls"] == 2, \
        "本套件mock: 抽取1+judge2=3次llm, 2次search; cost是本任务增量"
    rows = et.evaluate_batch([{"task": T_EXACT, "response": "答案: 8635亿"}], meter=m)
    assert rows[0]["cost"]["llm_calls"] == 0
    assert "cost" not in et.evaluate(T_EXACT, response="答案: 8635亿"), "不传meter无cost字段"


def test_summarize():
    m = ce.Meter()
    rows = [et.evaluate(T_EXACT, response="答案: 8635亿", meter=m),
            et.evaluate(T_EXACT, response="答案: 1亿", meter=m),
            et.evaluate(T_RETR, response="DOC", llm=mock_llm,
                        search=mock_search, meter=m)]
    s = et.summarize(rows)
    assert s["n"] == 3 and s["verifier_fp"] == rows[0]["verifier_fp"]
    ex = s["classes"]["exact"]
    assert ex["n"] == 2 and ex["mean_score"] == 0.5
    assert ex["cost"]["llm_calls"] == 0, "exact 类成本合计为零"
    rt = s["classes"]["retrieval"]
    assert rt["n"] == 1 and rt["cost"]["llm_calls"] == 3 and rt["cost"]["search_calls"] == 2
    # 混指纹: 默认拒绝, strict_fp=False 降级为列出
    fake = {**rows[0], "verifier_fp": "deadbeef0000"}
    try:
        et.summarize(rows + [fake])
        assert False, "混指纹必须拒绝汇总"
    except ValueError:
        pass
    s2 = et.summarize(rows + [fake], strict_fp=False)
    assert isinstance(s2["verifier_fp"], list) and len(s2["verifier_fp"]) == 2
    # None score(如全弃权)不进均值但计入 n
    none_row = {**rows[0], "score": None}
    s3 = et.summarize([none_row])
    assert s3["classes"]["exact"]["mean_score"] is None
    assert s3["classes"]["exact"]["n"] == 1 and s3["classes"]["exact"]["scored"] == 0


def test_repeat_evaluate():
    from paired_eval import rubric_eval as re_
    calls = []

    def flaky_llm(prompt, system, schema):   # 前4次met,后4次not_met
        assert system is re_.JUDGE_RUBRIC_SYSTEM
        calls.append(1)
        return {"verdict": "met" if len(calls) <= 4 else "not_met", "reasoning": "r"}

    t = {"id": "t-rep", "verification": {"class": "rubric",
         "criteria": [{"text": "条A", "weight": 1}]}}
    m = ce.Meter()
    rep = et.repeat_evaluate(t, n=8, response="RESP", llm=flaky_llm, meter=m)
    assert rep["scores"] == [1.0] * 4 + [0.0] * 4 and rep["mean_score"] == 0.5
    assert rep["successes"] == 4, "二值分数序列附带 successes 供 pass^k"
    assert abs(ce.pass_hat_k(rep["successes"], rep["n"], 2) - 3 / 14) < 1e-12
    assert all(r["cost"]["llm_calls"] == 1 for r in rep["runs"]), "每run成本独立归账"
    try:
        et.repeat_evaluate(t, n=0, response="RESP", llm=flaky_llm)
        assert False, "n<1 应拒绝"
    except ValueError:
        pass


def test_summarize_window_guard():
    """与 reliability_matrix 同源: 不同测量窗的结果不许并排汇总。"""
    rows = [et.evaluate(T_EXACT, response="答案: 8635亿"),
            et.evaluate(T_EXACT, response="答案: 1亿")]
    assert all(isinstance(r["measured_at"], float) for r in rows), "每行必须带测量时戳"
    assert et.summarize(rows)["n"] == 2, "同窗正常汇总"
    stale = dict(rows[0], measured_at=rows[0]["measured_at"] - 7200)
    try:
        et.summarize(rows + [stale])
        assert False, "跨2小时应报错"
    except ValueError as e:
        assert "漂移" in str(e)
    assert et.summarize(rows + [stale], max_span_s=None)["n"] == 3, "可显式关闭"
    # 无时戳的合成行不受影响(向后兼容),但混指纹仍拒绝
    synth = {"verification_class": "exact", "score": 1.0,
             "verifier_fp": rows[0]["verifier_fp"]}
    assert et.summarize([synth])["n"] == 1
    try:
        et.summarize([synth, dict(synth, verifier_fp="deadbeef")])
        assert False, "混指纹仍须拒绝"
    except ValueError:
        pass



def test_evaluate_pair_cancels_judge_order_bias():
    """交错判定: judge 的"先看到谁就给谁高分"这类顺序偏置必须两侧对消;
    真实质量差仍照常检出。"""
    from paired_eval import rubric_eval as re_
    seen = {"i": 0}

    def order_biased_judge(prompt, system, schema):
        assert system is re_.JUDGE_RUBRIC_SYSTEM
        seen["i"] += 1
        return {"verdict": "met" if seen["i"] % 2 == 1 else "not_met", "reasoning": "r"}

    t = {"id": "t-pair", "verification": {"class": "rubric",
         "criteria": [{"text": "条A", "weight": 1}]}}
    out = et.evaluate_pair(t, "回答A", "回答B", n=4, llm=order_biased_judge)
    assert [r["a_first"] for r in out["rows"]] == [True, False, True, False]
    assert out["a_mean"] == out["b_mean"] == 0.5, f"顺序偏置应对消: {out['a_mean']},{out['b_mean']}"
    assert out["compare"]["mean_diff"] == 0.0
    # 真实差异: judge 只认含关键词的回答
    kw_judge = lambda p, s, sch: {"verdict": "met" if "关键词" in p else "not_met", "reasoning": "r"}
    out2 = et.evaluate_pair(t, "含关键词的回答", "普通回答", n=2, llm=kw_judge)
    assert out2["a_mean"] == 1.0 and out2["b_mean"] == 0.0
    assert out2["compare"]["wins"] == 2
    # exact 类无需重复也可用; n<1 拒绝
    out3 = et.evaluate_pair(T_EXACT, "答案: 8635亿", "答案: 1亿")
    assert out3["a_mean"] == 1.0 and out3["b_mean"] == 0.0
    try:
        et.evaluate_pair(T_EXACT, "答案: 1", "答案: 2", n=0)
        assert False, "n<1 应拒绝"
    except ValueError:
        pass



def test_grounding_policy():
    """真但无据的 claim(参数知识)是否算违规,由任务声明决定 ——
    实测: 模型补一句事实正确但不在轨迹里的话, deep-research 该罚, 通用问答不该。"""
    def traj_llm(prompt, system, schema):
        if system is ce.EXTRACT_SYSTEM:
            return {"claims": [
                {"text": "有据事实A", "source_quote": "A", "verifiable": True,
                 "importance": "core", "search_query": "qA"},
                {"text": "有据事实B", "source_quote": "B", "verifiable": True,
                 "importance": "core", "search_query": "qB"},
                {"text": "真但无据的补充", "source_quote": "C", "verifiable": True,
                 "importance": "detail", "search_query": "qC"}]}
        if "无据" in prompt:
            return {"verdict": "fabricated", "source_tool_call": None,
                    "evidence_quote": "", "reasoning": "r"}
        return {"verdict": "grounded", "source_tool_call": "tc_1",
                "evidence_quote": "q", "reasoning": "r"}

    obs = [{"tool_call_id": "tc_1", "tool": "s", "observation": "o"}]
    strict = {"id": "t-strict", "verification": {"class": "trajectory"}}
    r1 = et.evaluate(strict, response="DOC", llm=traj_llm, observations=obs)
    assert abs(r1["score"] - 2 / 3) < 1e-9, "默认 must_ground: 无据即扣分"
    assert r1["metrics"]["grounding_policy"] == "must_ground"
    lenient = {"id": "t-lenient", "verification": {
        "class": "trajectory", "grounding_policy": "allow_parametric"}}
    r2 = et.evaluate(lenient, response="DOC", llm=traj_llm, observations=obs)
    assert r2["score"] == 1.0, "allow_parametric: 参数知识剔出分母"
    assert r2["metrics"]["parametric"] == 1 and r2["metrics"]["fabricated"] == 1, \
        "剔出分母但必须单独报数,不许消失"
    bad = {"id": "t-bad", "verification": {"class": "trajectory", "grounding_policy": "whatever"}}
    try:
        et.evaluate(bad, response="DOC", llm=traj_llm, observations=obs)
        assert False, "非法 policy 应拒绝"
    except ValueError:
        pass



def test_reason_propagates_to_summary():
    """grade_answer 的 reason 必须一路传到批次汇总: 单行有信号而报表看不见等于没有。
    (这条缺陷是本仓库自己造成过的 —— 新增字段后忘了在分发层传递。)"""
    rows = [
        et.evaluate(T_EXACT, response="答案: 8635亿"),                 # correct
        et.evaluate(T_EXACT, response="大约是八千多亿元吧"),            # no_slot
        et.evaluate(T_EXACT, response="   "),                          # blank
    ]
    assert rows[0]["details"]["reason"] is None
    assert rows[1]["details"]["reason"] == "no_slot"
    assert rows[2]["details"]["reason"] == "blank"
    s = et.summarize(rows)
    ex = s["classes"]["exact"]
    assert ex["non_attempt_reasons"] == {"no_slot": 1, "blank": 1}
    assert ex["n"] == 3 and ex["scored"] == 3, "弃答记0分仍计入,只是原因另有出口"
    # 无 reason 的类别不该凭空冒出该字段
    clean = et.summarize([rows[0]])
    assert clean["classes"]["exact"]["non_attempt_reasons"] is None



def test_repeat_reports_dispersion():
    """重复的目的就是量方差, 只报均值等于白重复:
    [1,0,1,0] 与恒定0.5 均值相同, 前者是判定不稳定, 后者是稳定中等分。"""
    from paired_eval import rubric_eval as re_
    flip = {"i": 0}

    def alternating(prompt, system, schema):
        assert system is re_.JUDGE_RUBRIC_SYSTEM
        flip["i"] += 1
        return {"verdict": "met" if flip["i"] % 2 else "not_met", "reasoning": "r"}

    def steady(prompt, system, schema):
        # 两条criteria各半: 每次都稳定得0.5
        return {"verdict": "met" if "条A" in prompt else "not_met", "reasoning": "r"}

    t1 = {"id": "t-flip", "verification": {"class": "rubric",
          "criteria": [{"text": "条A", "weight": 1}]}}
    t2 = {"id": "t-steady", "verification": {"class": "rubric",
          "criteria": [{"text": "条A", "weight": 1}, {"text": "条B", "weight": 1}]}}
    r1 = et.repeat_evaluate(t1, n=4, response="R", llm=alternating)
    r2 = et.repeat_evaluate(t2, n=4, response="R", llm=steady)
    assert r1["mean_score"] == r2["mean_score"] == 0.5, "均值相同"
    assert r1["score_min"] == 0.0 and r1["score_max"] == 1.0
    assert r2["score_min"] == r2["score_max"] == 0.5
    assert r1["score_stdev"] > 0.5 and r2["score_stdev"] == 0.0, \
        f"离散度必须区分两者: {r1['score_stdev']} vs {r2['score_stdev']}"
    assert r1["successes"] == 2, "二值序列仍附 successes"
    # n=1 时无标准差, 不硬造0
    single = et.repeat_evaluate(t2, n=1, response="R", llm=steady)
    assert single["score_stdev"] is None and single["scored"] == 1


def test_pair_dropped_reps_counted():
    """evaluate_pair 剔出无效轮次时必须计数并归因:
    静默丢弃会让 a_mean/b_mean 的分母悄悄变小, 报表看不出任何异常。"""
    from paired_eval import rubric_eval as re_
    state = {"i": 0}

    def sometimes_all_abstain(prompt, system, schema):
        assert system is re_.JUDGE_RUBRIC_SYSTEM
        state["i"] += 1
        # 第3、4次调用(第2轮的两侧)全弃权 -> 该轮两侧 score 均为 None
        return {"verdict": "abstain" if state["i"] in (3, 4) else "met", "reasoning": "r"}

    t = {"id": "t-pair-drop", "verification": {"class": "rubric",
         "criteria": [{"text": "条A", "weight": 1}]}}
    out = et.evaluate_pair(t, "回答A", "回答B", n=3, llm=sometimes_all_abstain)
    assert len(out["rows"]) == 3, "所有轮次都留痕"
    assert out["scored_reps"] == 2, "只有2轮进入比较"
    assert [d["rep"] for d in out["dropped_reps"]] == [1]
    assert set(out["dropped_reps"][0]["sides"]) == {"a", "b"}, "两侧都无分数"
    assert out["compare"]["n"] == 2, "配对检验只用有效轮次"
    assert out["a_mean"] == 1.0 and out["b_mean"] == 1.0
    # 无丢弃时为空列表
    steady = lambda p, s, sch: {"verdict": "met", "reasoning": "r"}
    clean = et.evaluate_pair(t, "A", "B", n=2, llm=steady)
    assert clean["dropped_reps"] == [] and clean["scored_reps"] == 2


def test_fingerprint_covers_every_prompt():
    """七个判定 prompt 每一个都必须影响指纹: 少算任何一个, 改动它就不会换尺子,
    而分数会被当成同尺可比。此前只测过改 JUDGE_WORLD_SYSTEM, 漏掉其余六个 ——
    从元组里删掉 rubric prompt 在变异测试中存活。"""
    from paired_eval import rubric_eval as re_
    base = et.verifier_fingerprint()
    targets = [(ce, "EXTRACT_SYSTEM"), (ce, "JUDGE_WORLD_SYSTEM"), (ce, "JUDGE_TRAJ_SYSTEM"),
               (ce, "DERIVED_SYSTEM"), (ce, "REFORMULATE_SYSTEM"), (ce, "CORROBORATE_SYSTEM"),
               (re_, "JUDGE_RUBRIC_SYSTEM")]
    for mod, name in targets:
        old = getattr(mod, name)
        try:
            setattr(mod, name, old + "MUTATED")
            assert et.verifier_fingerprint() != base, f"{name} 未参与指纹计算"
        finally:
            setattr(mod, name, old)
    assert et.verifier_fingerprint() == base, "全部还原后指纹必须回到原值"


def test_summarize_window_boundary():
    """跨度恰好等于上限时应放行(> 语义): 此前用例都是2小时 vs 1小时, 远离边界,
    于是 > 改成 >= 在变异测试中存活。"""
    row = et.evaluate(T_EXACT, response="答案: 8635亿")
    exact_edge = dict(row, measured_at=row["measured_at"] - 3600)
    s = et.summarize([row, exact_edge], max_span_s=3600)
    assert s["n"] == 2, "跨度恰好等于上限应放行"
    just_over = dict(row, measured_at=row["measured_at"] - 3600.001)
    try:
        et.summarize([row, just_over], max_span_s=3600)
        assert False, "略微超限必须拒绝"
    except ValueError:
        pass




def test_validate_task_rejects_malformed_task():
    """缺 id / 非 dict 的 task 此前从未被测: 机械变异把 or 改成 and 后存活,
    意味着"task 必须是含 id 的 dict"这条校验一直裸奔。"""
    for bad in [None, [], "task", 42, {}, {"verification": {"class": "exact", "gold": "1"}}]:
        try:
            et.validate_task(bad)
            assert False, f"畸形 task 应拒绝: {bad!r}"
        except ValueError:
            pass
    # 合法 task 仍通过
    v = et.validate_task(T_EXACT)
    assert v["class"] == "exact"




def test_pair_one_sided_none():
    """只有一侧无分数的轮次也必须被剔除并归因: 此前用例都是"两侧同时 None",
    于是把 and 改成 or 后仍然存活 —— 而 or 会让 None 混进均值计算直接崩。"""
    from paired_eval import rubric_eval as re_
    calls = {"i": 0}

    def b_abstains_first_rep(prompt, system, schema):
        assert system is re_.JUDGE_RUBRIC_SYSTEM
        calls["i"] += 1
        # 单条criteria: 第1轮 a_first -> 调用1=a侧, 调用2=b侧(令其弃权)
        return {"verdict": "abstain" if calls["i"] == 2 else "met", "reasoning": "r"}

    t = {"id": "t-one-side", "verification": {"class": "rubric",
         "criteria": [{"text": "条A", "weight": 1}]}}
    out = et.evaluate_pair(t, "回答A", "回答B", n=2, llm=b_abstains_first_rep)
    assert out["scored_reps"] == 1, f"仅第2轮可比: {out['scored_reps']}"
    assert [d["rep"] for d in out["dropped_reps"]] == [0]
    assert out["dropped_reps"][0]["sides"] == ["b"], "必须指出是 b 侧无分数"
    assert out["a_mean"] == 1.0 and out["b_mean"] == 1.0, "均值只用可比轮次"
    assert out["compare"]["n"] == 1




def test_all_routes_reject_missing_deps():
    """每条路由的依赖检查都要各自钉住: 此前只测过 retrieval 缺依赖,
    于是 trajectory 那行的 or 改成 and 后存活(缺单个依赖就不再报错)。"""
    obs = [{"tool_call_id": "tc_1", "tool": "s", "observation": "o"}]
    llm = lambda p, s, sch: {"verdict": "grounded", "source_tool_call": "tc_1",
                             "evidence_quote": "q", "reasoning": "r"}
    traj = {"id": "t-dep", "verification": {"class": "trajectory"}}
    rub = {"id": "r-dep", "verification": {"class": "rubric",
           "criteria": [{"text": "条A", "weight": 1}]}}
    cases = [
        (traj, {"response": "DOC", "llm": llm}),                    # 缺 observations
        (traj, {"response": "DOC", "observations": obs}),            # 缺 llm
        (traj, {"llm": llm, "observations": obs}),                   # 缺 response
        (rub, {"response": "DOC"}),                                  # 缺 llm
        (rub, {"llm": llm}),                                         # 缺 response
    ]
    for task, kw in cases:
        try:
            et.evaluate(task, **kw)
            assert False, f"{task['id']} 缺依赖却未报错: {sorted(kw)}"
        except ValueError:
            pass


def test_repeat_edge_cases():
    """两个边界: n=2 必须有标准差(> 1 改成 > 2 后存活);
    全部分数为 None 时不得凭空报出 successes(and 改成 or 后存活)。"""
    from paired_eval import rubric_eval as re_
    flip = {"i": 0}

    def alternating(prompt, system, schema):
        flip["i"] += 1
        return {"verdict": "met" if flip["i"] % 2 else "not_met", "reasoning": "r"}

    t = {"id": "t-n2", "verification": {"class": "rubric",
         "criteria": [{"text": "条A", "weight": 1}]}}
    r2 = et.repeat_evaluate(t, n=2, response="R", llm=alternating)
    assert r2["score_stdev"] is not None and r2["score_stdev"] > 0, \
        f"n=2 应给出标准差: {r2['score_stdev']!r}"
    # 全弃权 -> 每轮 score 均为 None
    all_abstain = lambda p, s, sch: {"verdict": "abstain", "reasoning": "r"}
    r0 = et.repeat_evaluate(t, n=3, response="R", llm=all_abstain)
    assert r0["scores"] == [] and r0["mean_score"] is None
    assert "successes" not in r0, "没有二值结果时不得报出 successes"
    assert r0["score_stdev"] is None and r0["scored"] == 0


def _rubric_llm_counting():
    """rubric 评委: 记录调用次数, 全部判 met。"""
    calls = {"n": 0}

    def llm(prompt, system, schema):
        assert system is re_.JUDGE_RUBRIC_SYSTEM
        calls["n"] += 1
        return {"verdict": "met", "reasoning": "r"}
    return llm, calls


T_GATED = {"id": "t-gated", "instruction": "算 8635 亿并说明",
           "verification": {"class": "gated",
                            "gate": {"class": "exact", "gold": "8635亿", "kind": "numeric"},
                            "score": {"class": "rubric",
                                      "criteria": [{"text": "给出了计算过程", "weight": 2},
                                                   {"text": "单位正确", "weight": 1}]}}}


def test_gated_fails_gate_scores_zero_and_skips_judge():
    """"能验的先验, 验不过直接 0 分": gate 不过时 score 侧根本不跑 —— 不为已经错了的回答付 judge 的钱,
    也不给它一个"质量分"。这是最初那句"有真值用程序验证, 没有用 rubric"最直接的形态。"""
    llm, calls = _rubric_llm_counting()
    r = et.evaluate(T_GATED, response="答案: 9000亿, 因为...", llm=llm)
    assert r["score"] == 0.0 and r["verdict"] == "gated_out"
    assert r["metrics"]["gate_passed"] is False and r["metrics"]["score_evaluated"] is False
    assert r["metrics"]["gate_score"] == 0.0 and r["metrics"]["gate_min"] == 1.0
    assert r["metrics"]["gate_class"] == "exact" and r["metrics"]["score_class"] == "rubric"
    assert r["details"]["gate"]["verdict"] == "incorrect" and r["details"]["score"] is None
    assert calls["n"] == 0, "gate 不过, judge 一次都不该被调用"
    assert r["verification_class"] == "gated" and r["task_id"] == "t-gated"


def test_gated_passes_gate_then_rubric_scores():
    llm, calls = _rubric_llm_counting()
    r = et.evaluate(T_GATED, response="答案: 8635亿, 由 A+B 得到", llm=llm)
    assert r["score"] == 1.0 and r["verdict"] is None
    assert r["metrics"]["gate_passed"] is True and r["metrics"]["score_evaluated"] is True
    assert r["metrics"]["score_metrics"]["judged_weight_share"] == 1.0
    assert r["details"]["gate"]["task_id"] == "t-gated.gate" and r["details"]["score"]["task_id"] == "t-gated.score"
    assert calls["n"] == 2, "两条 criteria -> 两次 judge 调用"
    # 部分 met: 分数来自 rubric 加权
    def half(prompt, system, schema):
        return {"verdict": "met" if "计算过程" in prompt else "not_met", "reasoning": "r"}
    r2 = et.evaluate(T_GATED, response="答案: 8635亿", llm=half)
    assert abs(r2["score"] - 2 / 3) < 1e-9, r2["score"]


def test_gated_gate_min_with_continuous_gate():
    """gate 可以是连续分(如 trajectory 的 grounding_rate): gate_min 定"过"的线, 默认 1.0。"""
    task = {"id": "t-gt", "instruction": "总结",
            "verification": {"class": "gated", "gate_min": 0.5,
                             "gate": {"class": "trajectory", "grounding_policy": "must_ground"},
                             "score": {"class": "rubric", "criteria": [{"text": "简洁", "weight": 1}]}}}

    def llm(prompt, system, schema):
        if system is ce.EXTRACT_SYSTEM:
            return mock_llm(prompt, system, schema)           # 两条 claim: 一真一假 -> grounding 0.5
        if system is ce.JUDGE_TRAJ_SYSTEM:
            return mock_llm(prompt, system, schema)
        assert system is re_.JUDGE_RUBRIC_SYSTEM
        return {"verdict": "met", "reasoning": "r"}
    r = et.evaluate(task, response="K真事实。W假陈述。", observations=OBS, llm=llm)
    assert r["metrics"]["gate_score"] == 0.5 and r["metrics"]["gate_passed"] is True and r["score"] == 1.0
    task["verification"]["gate_min"] = 0.6
    r2 = et.evaluate(task, response="K真事实。W假陈述。", observations=OBS, llm=llm)
    assert r2["verdict"] == "gated_out" and r2["score"] == 0.0


def test_gated_validation():
    base = {"class": "exact", "gold": "1", "kind": "numeric"}
    rub = {"class": "rubric", "criteria": [{"text": "x", "weight": 1}]}
    bad = [
        ({"class": "gated", "score": rub}, "必须提供 gate"),
        ({"class": "gated", "gate": base}, "必须提供 score"),
        ({"class": "gated", "gate": {"class": "gated", "gate": base, "score": rub}, "score": rub}, "不得再嵌套"),
        ({"class": "gated", "gate": {"class": "exact"}, "score": rub}, "exact 类必须提供 gold"),
        ({"class": "gated", "gate": base, "score": {"class": "rubric", "criteria": []}}, "非空 criteria"),
        ({"class": "gated", "gate": base, "score": rub, "gate_min": 0}, "gate_min"),
        ({"class": "gated", "gate": base, "score": rub, "gate_min": 1.5}, "gate_min"),
        ({"class": "gated", "gate": base, "score": rub, "gate_min": True}, "gate_min"),
    ]
    for v, msg in bad:
        try:
            et.validate_task({"id": "x", "verification": v})
            raise AssertionError(f"应报错: {v}")
        except ValueError as e:
            assert msg in str(e), (msg, str(e))
    assert et.validate_task({"id": "x", "verification": {"class": "gated", "gate": base, "score": rub}})["class"] == "gated"


def test_gated_cost_covers_both_stages():
    """meter 计的是整个 gated 任务的成本: gate 过时含 judge 调用, 不过时为 0。"""
    m = ce.Meter()
    llm, _ = _rubric_llm_counting()
    r_pass = et.evaluate(T_GATED, response="答案: 8635亿", llm=llm, meter=m)
    assert r_pass["cost"]["llm_calls"] == 2
    r_fail = et.evaluate(T_GATED, response="答案: 1", llm=llm, meter=m)
    assert r_fail["cost"]["llm_calls"] == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
