#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整条方法学流水线的集成测试: 筛题 -> 交错 A/B -> 完整报告 -> 结论翻译。

为什么单独一支: 现有测试都在验单个组件, 而组件之间的接口一致性无人检查 ——
report 报的单元数是否等于 run_interleaved 实际跑的轮次? saturation 数出的有信息题
是否与 screen_graded 的保留题对得上? interpret 的 n_units 是否与 McNemar 用的
不一致对同源? 这类"各自都对, 合起来错"的缺口只有端到端跑一遍才暴露。

全部用确定性假模型, 不碰真实调用, 故属第 1 层快速套件。
"""
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 项目根: 让 `python3 tests/x.py` 直接可跑
from paired_eval import claim_eval as ce
from paired_eval import paired_bench as pb


# 三类候选: 中段(有余量)、恒过、恒败 —— 覆盖筛选的三条出口
_RATES = {"mid-a": 0.5, "mid-b": 0.6, "top": 1.0, "bottom": 0.0}


def _cyclic_grader():
    """确定性 grader: 按固定序列出分, 使任意连续 2 轮的均分都落在目标区间。
    序列必须让"前 n 轮"与"整体均分"同侧 —— 否则 fixture 自己就会踩到
    docs/lessons.md#screening-reliability 里的坑: 周期与轮数不互质时, 中段题会因前几轮凑巧满分而被判恒过。"""
    seqs = {"mid-a": [1.0, 0.0], "mid-b": [1.0, 0.0, 1.0, 0.0, 0.0],
            "top": [1.0], "bottom": [0.0]}
    pos = {k: 0 for k in seqs}

    def grade(case):
        k = case["id"]
        v = seqs[k][pos[k] % len(seqs[k])]
        pos[k] += 1
        return v

    return grade


def test_screen_then_ab_then_report_composes():
    """全流水线: 筛出中段题 -> 在其上做 A/B -> 报告 -> 翻译成结论。
    每一步的输出都是下一步的输入, 断言集中在跨组件的一致性上。"""
    cases = [{"id": k} for k in ("mid-a", "mid-b", "top", "bottom")]
    scr = pb.screen_graded(cases, _cyclic_grader(), n=2, confirm_n=2, band=(0.0, 0.9))
    kept = [c["id"] for c in scr["kept"]]
    assert kept == ["mid-a", "mid-b"], f"筛选应只留中段题: {scr}"
    assert scr["saturated_pass"] == ["top"] and scr["saturated_fail"] == ["bottom"]

    # 用筛出的题做 A/B。真题需要 check, 故从 ALL_TASKS 取同样数量的题代入
    tasks = [t for t in pb.ALL_TASKS if t["id"] in pb.SCAFFOLD_SENSITIVE][:len(kept)]
    assert len(tasks) == 2, "敏感子集须至少 2 题, 否则流水线测不到多题聚合"
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    look = lambda p: canon[next(k for k in canon if k in p)]
    # A 全对; B 在第一题上恒错 -> 不一致对全部来自该题, 集中度应为 1.0
    good = lambda p: look(p)
    bad_first = lambda p: ("错" if tasks[0]["instruction"] in p else look(p))

    n = 5
    out = pb.run_interleaved({"A": good, "B": bad_first}, tasks=tasks, n=n)
    rp = pb.report(out["reports"], refusals=out["refusals"])
    pair = rp["pairs"][0]

    # 跨组件一致性 1: report 的单元数 == 题数 x 轮数
    assert pair["units"] == len(tasks) * n == 10, pair["units"]
    # 跨组件一致性 2: interpret 拿到的 n_units 与 report 算的一致
    assert pair["interpretation"]["n_units"] == pair["units"]
    # 跨组件一致性 3: saturation 的有信息题数 == 真正出现分歧的题数
    assert rp["saturation"]["informative"] == 1, rp["saturation"]
    assert rp["saturation"]["ids"]["informative"] == [tasks[0]["id"]]
    assert rp["saturation"]["ids"]["saturated_pass"] == [tasks[1]["id"]]
    # 跨组件一致性 4: 不一致对总数 == 该题上 A 胜 B 的轮次数
    a_only, b_only = pair["a_only"], pair["b_only"]
    assert (a_only, b_only) == (n, 0), f"A 应在第一题上每轮都胜: {pair}"
    # 跨组件一致性 5: by_task 的逐题分解必须精确加总回总数(否则集中度是错的)
    by = {r["id"]: (r["a_only"], r["b_only"]) for r in pair["by_task"]}
    assert sum(x for x, _ in by.values()) == a_only, f"逐题 a_only 之和须等于总数: {by}"
    assert sum(y for _, y in by.values()) == b_only, f"逐题 b_only 之和须等于总数: {by}"
    assert by[tasks[0]["id"]] == (n, 0), f"分歧全在第一题: {by}"
    assert by.get(tasks[1]["id"], (0, 0)) == (0, 0), "恒过题不该有不一致对"
    # 跨组件一致性 6: 集中度 = 最大单题贡献 / 总不一致对 -> 全来自一题时为 1.0
    assert pair["concentration"] == 1.0, pair["concentration"]
    assert max(x + y for x, y in by.values()) / (a_only + b_only) == pair["concentration"]

    # 结论翻译 —— 这里是本测试最有价值的一条: 单元数(10)看起来够, 地板 0.002,
    # 但 McNemar 只用得上 5 个不一致对, 可达最小 p 是 0.0625 -> 到不了显著。
    # 集成测试当初正是在这里抓到 interpret 用错基数(报"未检出差异, 只能排除 >=67%")。
    assert ce.p_floor(pair["units"]) < 0.05, "按单元算地板很低 —— 这正是误导所在"
    assert ce.p_floor(a_only + b_only) == 0.0625, "按不一致对算才是真地板"
    v = pair["interpretation"]
    assert v["verdict"] == "null" and "检验无力" in v["text"], v
    assert v["p_floor"] == 0.0625 and "不一致对只有 5 个" in v["text"]
    assert "需至少 6 个不一致对" in v["text"], "处方要指向不一致对, 不是加轮数"

    # 加一轮把不一致对推到 6 个: 同样的效应此时才够显著
    out6 = pb.run_interleaved({"A": good, "B": bad_first}, tasks=tasks, n=6)
    p6 = pb.report(out6["reports"], refusals=out6["refusals"])["pairs"][0]
    assert p6["a_only"] + p6["b_only"] == 6
    assert p6["interpretation"]["verdict"] == "significant", p6["interpretation"]
    assert (p6["mean_diff"] > 0) == (p6["a"] == "A"), "方向必须与更强的系统一致"

    # 报告文本必须自带全部诊断项 —— 少一项就可能让读者据不全的信息下结论
    for token in ("有效样本", "拒答", "Δ=", "CI95", "逐题p", "逐轮McNemar",
                  "Holm", "不一致对", "集中度"):
        assert token in rp["text"], f"报告缺 {token}"
    # 触顶诊断: A 全对 -> 触顶(单系统触顶的措辞是"作为参照时效应被压缩", 非"不可解释")
    assert rp["ceiling"] == {"at_top": ["A"], "at_bottom": []}, rp["ceiling"]
    assert "作为参照时效应会被压缩" in rp["text"]
    assert "交互项都不可解释" not in rp["text"], "只有一个系统触顶时不该用多系统措辞"


def test_pipeline_underpowered_path_stays_honest():
    """同一条流水线在样本不足时必须给出"检验无力"而非"无差异" ——
    这是第107/115轮两次踩坑的组合: 少轮次 + 大效应, 最容易被误读成 null。"""
    tasks = [t for t in pb.ALL_TASKS if t["id"] in pb.SCAFFOLD_SENSITIVE][:2]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    look = lambda p: canon[next(k for k in canon if k in p)]
    out = pb.run_interleaved({"A": lambda p: look(p), "B": lambda p: "错"},
                             tasks=tasks, n=1)
    rp = pb.report(out["reports"], refusals=out["refusals"])
    pair = rp["pairs"][0]
    assert pair["units"] == 2
    v = pair["interpretation"]
    assert v["verdict"] == "null" and "检验无力" in v["text"], v
    # 关键: 地板按不一致对(2 对)算而非单元数(2) —— 此例两者巧合相等, 但处方不同:
    # "需 6 个不一致对"意味着要更能拉开差距的题, 而非单纯加轮数
    assert v["p_floor"] == 0.5, v
    assert "不一致对只有 2 个" in v["text"] and "需至少 6 个不一致对" in v["text"], v
    # 效应是压倒性的(B 全错), 但结论仍不许声称显著 —— 且点估计要照实报出
    assert abs(pair["mean_diff"]) == 1.0
    assert "1.000" in v["text"], "无力时仍须给点估计, 否则读者以为没效应"
    # 触底诊断: B 全错 -> 应被标为触底
    assert rp["ceiling"]["at_bottom"] == ["B"], rp["ceiling"]


def test_pipeline_rejects_incoherent_inputs():
    """流水线的守卫必须在最早的环节拦住坏输入, 而不是让坏数据流到报告里。
    实测确认守卫比预期更早: 单系统在 run_interleaved 就被拒, 不必等到 report。"""
    tasks = [t for t in pb.ALL_TASKS if t["id"] in pb.SCAFFOLD_SENSITIVE][:2]
    canon = {t["instruction"]: t["canonical"] for t in tasks}
    look = lambda p: canon[next(k for k in canon if k in p)]
    # 单系统: 最早的环节(交错运行)就拒 —— 配对比较无从谈起
    try:
        pb.run_interleaved({"only": look}, tasks=tasks, n=2)
        raise AssertionError("单系统应在 run_interleaved 就被拒")
    except ValueError as e:
        assert "两个系统" in str(e), str(e)
    # 即便绕过运行层直接喂给 report, 也必须拒 —— 守卫不能只在入口
    fake = {"only": [{"id": t["id"], "n": 1, "successes": 1, "runs": [True],
                      "pass_at_1": 1.0, "pass_hat_k": 1.0} for t in tasks]}
    try:
        pb.report(fake, require_interleaved=False)
        raise AssertionError("单系统喂给 report 也应被拒")
    except (ValueError, IndexError, KeyError):
        pass
    # 空题集
    try:
        pb.run_interleaved({"A": look, "B": look}, tasks=[], n=2)
        raise AssertionError("空题集应被拒")
    except (ValueError, IndexError, KeyError):
        pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
