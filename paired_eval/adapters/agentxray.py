# -*- coding: utf-8 -*-
"""Turn an AgentXRay session export into paired-eval trajectory inputs.

AgentXRay (https://github.com/alloevil/AgentXRay) normalises Claude Code / Codex / OpenClaw /
Hermes / OMP / dsh / Gemini CLI logs into one shape, served at
``GET /api/<platform>/sessions/<id>`` as ``{"session": {...}, "messages": [...]}``.
Each message has ``role`` in {user, assistant, toolCall, toolResult, system}, ``content`` as a list
of ``{"type": "text", "text": ...}`` blocks (Hermes additionally embeds ``{"type": "toolCall", ...}``
blocks inside assistant content), plus ``toolCallId`` / ``toolName`` on the tool messages.

This module is pure standard library and does no I/O of its own: hand it the parsed JSON.

    import json, paired_eval as pe
    from paired_eval.adapters.agentxray import trajectory_task, observations, final_answer

    sess = json.load(open("session.json"))          # curl http://localhost:3800/api/codex/sessions/<id>
    task = trajectory_task(sess, grounding_policy="must_ground")
    r = pe.evaluate(task, response=final_answer(sess), observations=task["observations"], llm=judge)
    r["score"]                                       # grounding rate of the agent's final answer

Two agent runs on the same instruction become one paired unit: build a task from each, evaluate,
and feed the per-task scores to ``pe.paired_compare`` — that is the "agent" axis in the README.
"""

__all__ = ["observations", "final_answer", "instruction", "trajectory_task"]


def _text(message):
    """Concatenate the text blocks of a normalised message."""
    parts = []
    for block in message.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts)


def _tool_names(messages):
    """tool_call_id -> tool name, from toolCall messages and from toolCall blocks embedded in
    assistant content (Hermes shape). toolResult rows that carry their own toolName win later."""
    names = {}
    for m in messages:
        if m.get("role") == "toolCall" and m.get("toolCallId"):
            names[m["toolCallId"]] = m.get("toolName")
        for block in m.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "toolCall" and block.get("id"):
                names[block["id"]] = block.get("name")
    return names


def observations(session, max_chars=None):
    """The agent's tool results as paired-eval observations:
    ``[{"tool_call_id", "tool", "observation"}, ...]`` in session order.

    A toolResult without a toolCallId gets a positional id (``obs-<index>``) so the judge can still
    cite it. ``max_chars`` truncates each observation (long file dumps drown the judge prompt)."""
    messages = session.get("messages") or []
    names = _tool_names(messages)
    out = []
    for i, m in enumerate(messages):
        if m.get("role") != "toolResult":
            continue
        text = _text(m)
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + "\u2026"
        call_id = m.get("toolCallId") or f"obs-{i}"
        out.append({
            "tool_call_id": call_id,
            "tool": m.get("toolName") or names.get(call_id) or "tool",
            "observation": text,
        })
    return out


def final_answer(session):
    """Text of the last assistant message that has any text — what the user actually read."""
    for m in reversed(session.get("messages") or []):
        if m.get("role") == "assistant":
            text = _text(m)
            if text.strip():
                return text
    return ""


def instruction(session):
    """Text of the first user message: the task the agent was given."""
    for m in session.get("messages") or []:
        if m.get("role") == "user":
            text = _text(m)
            if text.strip():
                return text
    return ""


def trajectory_task(session, task_id=None, grounding_policy="must_ground", max_chars=None):
    """A paired-eval task dict of class ``trajectory`` for this session.

    ``observations`` is included on the task (as the README examples do) so ``evaluate`` can be
    called with ``observations=task["observations"]``. ``task_id`` defaults to the session id."""
    sess = session.get("session") or {}
    return {
        "id": task_id or sess.get("id") or "agentxray-session",
        "instruction": instruction(session),
        "observations": observations(session, max_chars=max_chars),
        "verification": {"class": "trajectory", "grounding_policy": grounding_policy},
    }
