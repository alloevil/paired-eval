# -*- coding: utf-8 -*-
"""OpenAI 兼容接口的最小适配器(仅标准库): 把一个 HTTP 端点接成本仓库要求的两种注入签名。

    call(prompt) -> str                    # paired_bench.make_model / reproduce_findings.main 用
    llm(prompt, system, schema) -> dict    # claim_eval / eval_task / rubric_eval 的 judge 用

本仓库不绑定任何供应商, 所以这个文件放在 examples/ 而不是库里: 它只是"怎么接"的一个示范,
任何满足上面两个签名的可调用对象都能替代它。配置来自参数或环境变量:
    OPENAI_BASE_URL (默认 https://api.openai.com/v1)  OPENAI_API_KEY  OPENAI_MODEL

用法:
    from examples.adapter_openai_compat import make_call, make_llm
    call = make_call(model="gpt-4o-mini")
    llm = make_llm(model="gpt-4o-mini")
    import reproduce_findings as rf; rf.main(call, judge=llm)
"""
import json
import os
import re
import urllib.error
import urllib.request

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def _config(model, base_url, api_key):
    cfg = {"model": model or os.environ.get("OPENAI_MODEL"),
           "base_url": (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
           "api_key": api_key or os.environ.get("OPENAI_API_KEY")}
    missing = [k for k in ("model", "api_key") if not cfg[k]]
    if missing:
        raise ValueError(f"缺少配置 {missing}: 传参或设置环境变量 "
                         f"{', '.join('OPENAI_' + m.upper() for m in missing)}")
    return cfg


def _chat(cfg, body, timeout):
    """POST /chat/completions, 返回首条 message.content。非 2xx 抛 RuntimeError 并带响应片段 ——
    这里刻意不吞错: paired_bench.make_model 负责有界重试, 适配器只负责如实报告。"""
    req = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        snippet = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {e.code} from {cfg['base_url']}: {snippet}") from None
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"响应缺少 choices[0].message.content: {json.dumps(payload)[:300]}") from None


def make_call(model=None, base_url=None, api_key=None, timeout=60, temperature=0.0):
    """裸文本调用。temperature 默认 0: 评测要的是可复现, 不是多样性。"""
    cfg = _config(model, base_url, api_key)

    def call(prompt):
        return _chat(cfg, {"model": cfg["model"], "temperature": temperature,
                           "messages": [{"role": "user", "content": prompt}]}, timeout)
    return call


def make_llm(model=None, base_url=None, api_key=None, timeout=90, temperature=0.0,
             json_mode="schema"):
    """结构化调用。json_mode 决定怎么把 schema 交给服务端:
        "schema"  response_format=json_schema(strict)   —— OpenAI 及多数兼容服务
        "object"  response_format=json_object + schema 写进 system —— 只支持 JSON 模式的服务
        "prompt"  不传 response_format, schema 写进 system          —— 完全不支持的服务
    返回值一律 json.loads 后的 dict; 服务端若包了 ```json 围栏也会被剥掉(不少兼容服务会这样)。"""
    if json_mode not in ("schema", "object", "prompt"):
        raise ValueError(f"json_mode 只能是 schema / object / prompt, 得到 {json_mode!r}")
    cfg = _config(model, base_url, api_key)

    def llm(prompt, system, schema):
        sys_text = system or ""
        if json_mode != "schema":
            sys_text = (sys_text + "\n\n只输出一个满足以下 JSON schema 的 JSON 对象, 不要其他内容:\n"
                        + json.dumps(schema, ensure_ascii=False)).strip()
        messages = ([{"role": "system", "content": sys_text}] if sys_text else []) + \
                   [{"role": "user", "content": prompt}]
        body = {"model": cfg["model"], "temperature": temperature, "messages": messages}
        if json_mode == "schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "out", "schema": schema, "strict": True}}
        elif json_mode == "object":
            body["response_format"] = {"type": "json_object"}
        text = _chat(cfg, body, timeout)
        m = _FENCE.match(text)
        try:
            out = json.loads(m.group(1) if m else text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"服务端返回的不是合法 JSON: {e.msg} @ {e.pos}: {text[:200]!r}") from None
        if not isinstance(out, dict):
            raise RuntimeError(f"期望 JSON 对象, 得到 {type(out).__name__}: {text[:200]!r}")
        return out
    return llm
