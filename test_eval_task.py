# -*- coding: utf-8 -*-
"""eval_task 离线测试:mock llm/search,零网络。运行: python3 test_eval_task.py"""
import claim_eval as ce
import eval_task as et


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
    import rubric_eval as re_

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
    import rubric_eval as re_
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
    import rubric_eval as re_
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



if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
