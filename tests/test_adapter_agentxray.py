# -*- coding: utf-8 -*-
"""agentxray 适配器: 把 AgentXRay 的规范化会话变成 trajectory 任务。
fixture 是 AgentXRay 自己的测试夹具经其 HTTP API 导出的真实响应(codex 平台), 不是手写的。
运行: python3 tests/test_adapter_agentxray.py"""
import json
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from paired_eval import claim_eval as ce
from paired_eval import eval_task as et
from paired_eval.adapters import agentxray as ax

ROOT = _pathlib.Path(__file__).resolve().parent
SESSION = json.loads((ROOT / "fixtures" / "agentxray-codex-session.json").read_text(encoding="utf-8"))


def test_observations_come_from_tool_results_with_ids_and_names():
    obs = ax.observations(SESSION)
    assert len(obs) == 1
    assert obs[0]["tool_call_id"] == "call-fx-1"
    assert obs[0]["tool"] == "shell", "toolResult 没带 toolName 时要从配对的 toolCall 取"
    assert obs[0]["observation"], "observation 是工具结果正文"


def test_instruction_and_final_answer_pick_the_right_turns():
    assert ax.instruction(SESSION).startswith("fixture: search-needle-alpha")
    last = ax.final_answer(SESSION)
    assert last and last == "\n".join(
        b["text"] for b in SESSION["messages"][-1]["content"] if b["type"] == "text")


def test_task_validates_and_routes_through_evaluate():
    task = ax.trajectory_task(SESSION)
    et.validate_task(task)
    assert task["id"] == SESSION["session"]["id"]
    assert task["verification"] == {"class": "trajectory", "grounding_policy": "must_ground"}
    seen = {}

    def llm(prompt, system, schema):
        if system is ce.EXTRACT_SYSTEM:
            return {"claims": [{"text": "claim-A", "source_quote": "A", "verifiable": True,
                                "importance": "core", "search_query": "A"}]}
        assert system is ce.JUDGE_TRAJ_SYSTEM
        seen["prompt"] = prompt
        return {"verdict": "grounded", "source_tool_call": "call-fx-1", "evidence_quote": "q", "reasoning": "r"}

    r = et.evaluate(task, response=ax.final_answer(SESSION), observations=task["observations"], llm=llm)
    assert r["score"] == 1.0
    assert "[call-fx-1] (shell)" in seen["prompt"], "judge 看到的 observation 带 tool_call_id 与工具名"


def test_hermes_shape_toolcall_blocks_inside_assistant_content():
    sess = {"session": {"id": "h1"}, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "do it"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                                          {"type": "toolCall", "id": "c1", "name": "terminal", "arguments": {}}]},
        {"role": "toolResult", "toolCallId": "c1", "toolName": None, "content": [{"type": "text", "text": "a\nb"}]},
        {"role": "toolResult", "toolCallId": None, "content": [{"type": "text", "text": "x" * 50}]},
        {"role": "assistant", "content": [{"type": "text", "text": "two files"}]},
    ]}
    obs = ax.observations(sess, max_chars=10)
    assert [o["tool_call_id"] for o in obs] == ["c1", "obs-3"], "无 id 的结果用位置 id, 仍可被引用"
    assert obs[0]["tool"] == "terminal"
    assert obs[1]["observation"] == "x" * 10 + "\u2026"
    assert ax.final_answer(sess) == "two files"


def test_empty_session_yields_valid_empty_task():
    task = ax.trajectory_task({"session": {}, "messages": []})
    et.validate_task(task)
    assert task["observations"] == [] and task["instruction"] == "" and task["id"] == "agentxray-session"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
