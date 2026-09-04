#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""examples/adapter_openai_compat 的测试: 起一个真实的本地 HTTP 服务, 走完整的 urllib 路径。

不 mock urllib —— 适配器的全部价值就在请求体、头、错误处理这几处线上细节, mock 掉就没剩什么可测。
服务端是可编程的假 OpenAI: 记录收到的请求, 按预设返回内容或状态码。
"""
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 项目根: 让 `python3 tests/x.py` 直接可跑
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from examples import adapter_openai_compat as ad


class _FakeOpenAI:
    """可编程假服务: next_content 决定返回什么, next_status 决定状态码, requests 记录收到的一切。"""

    def __init__(self):
        self.requests = []
        self.next_content = "ok"
        self.next_status = 200
        self.raw_body = None      # 非 None 时原样返回(用于测畸形响应)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                       "body": body})
                if outer.raw_body is not None:
                    data = outer.raw_body
                else:
                    data = json.dumps({"choices": [{"message": {"content": outer.next_content}}]}
                                      if outer.next_status == 200 else
                                      {"error": {"message": "boom", "type": "server_error"}}).encode("utf-8")
                self.send_response(outer.next_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):   # 静音
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02},
                         daemon=True).start()   # 默认 0.5s 轮询会让 shutdown() 拖慢整个快速套件

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _with_server(fn):
    srv = _FakeOpenAI()
    try:
        return fn(srv)
    finally:
        srv.close()


def test_call_sends_expected_request_and_returns_content():
    def body(srv):
        srv.next_content = "第一行\n第二行"
        call = ad.make_call(model="m1", base_url=srv.base_url, api_key="sk-test")
        assert call("你好") == "第一行\n第二行"
        req = srv.requests[-1]
        assert req["path"] == "/v1/chat/completions"
        assert req["auth"] == "Bearer sk-test"
        assert req["body"]["model"] == "m1" and req["body"]["temperature"] == 0.0
        assert req["body"]["messages"] == [{"role": "user", "content": "你好"}]
        assert "response_format" not in req["body"], "裸文本调用不该带 response_format"
        # base_url 末尾多一个 / 也要拼对
        call2 = ad.make_call(model="m1", base_url=srv.base_url + "/", api_key="k")
        call2("x")
        assert srv.requests[-1]["path"] == "/v1/chat/completions"
    _with_server(body)


def test_llm_three_json_modes():
    schema = {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]}

    def body(srv):
        srv.next_content = json.dumps({"verdict": "grounded"})
        # schema 模式: response_format=json_schema(strict), schema 原样透传, system 不被改写
        llm = ad.make_llm(model="m", base_url=srv.base_url, api_key="k", json_mode="schema")
        assert llm("判定", "你是评委", schema) == {"verdict": "grounded"}
        b = srv.requests[-1]["body"]
        assert b["response_format"]["type"] == "json_schema"
        assert b["response_format"]["json_schema"]["schema"] is not None
        assert b["response_format"]["json_schema"]["schema"] == schema
        assert b["response_format"]["json_schema"]["strict"] is True
        assert b["messages"][0] == {"role": "system", "content": "你是评委"}
        assert b["messages"][1] == {"role": "user", "content": "判定"}
        # object 模式: json_object + schema 写进 system
        llm_o = ad.make_llm(model="m", base_url=srv.base_url, api_key="k", json_mode="object")
        llm_o("判定", "你是评委", schema)
        b = srv.requests[-1]["body"]
        assert b["response_format"] == {"type": "json_object"}
        assert b["messages"][0]["role"] == "system" and "你是评委" in b["messages"][0]["content"]
        assert '"verdict"' in b["messages"][0]["content"], "schema 必须写进 system 提示"
        # prompt 模式: 无 response_format, schema 仍写进 system; system 为空时也要生成 system 消息
        llm_p = ad.make_llm(model="m", base_url=srv.base_url, api_key="k", json_mode="prompt")
        llm_p("判定", "", schema)
        b = srv.requests[-1]["body"]
        assert "response_format" not in b
        assert b["messages"][0]["role"] == "system" and "JSON schema" in b["messages"][0]["content"]
        # schema 模式下 system 为空: 不生成空 system 消息
        llm("判定", "", schema)
        assert srv.requests[-1]["body"]["messages"][0]["role"] == "user"
    _with_server(body)


def test_llm_strips_code_fences_and_rejects_non_objects():
    schema = {"type": "object"}

    def body(srv):
        llm = ad.make_llm(model="m", base_url=srv.base_url, api_key="k")
        for fenced in ('```json\n{"a": 1}\n```', '```\n{"a": 1}\n```', '  {"a": 1}  '):
            srv.next_content = fenced
            assert llm("p", None, schema) == {"a": 1}, fenced
        # 合法 JSON 但不是对象 -> 明确报错, 不许静默返回 list
        srv.next_content = "[1, 2]"
        try:
            llm("p", None, schema)
            raise AssertionError("非对象应报错")
        except RuntimeError as e:
            assert "期望 JSON 对象" in str(e) and "list" in str(e)
        # 非 JSON -> 报错里带原文片段, 便于排查
        srv.next_content = "抱歉, 我无法判断。"
        try:
            llm("p", None, schema)
            raise AssertionError("非 JSON 应报错")
        except RuntimeError as e:
            assert "不是合法 JSON" in str(e) and "抱歉" in str(e)
    _with_server(body)


def test_http_error_and_malformed_payload_are_reported_not_swallowed():
    def body(srv):
        call = ad.make_call(model="m", base_url=srv.base_url, api_key="k")
        srv.next_status = 429
        try:
            call("p")
            raise AssertionError("429 应抛错")
        except RuntimeError as e:
            assert "HTTP 429" in str(e) and "boom" in str(e), str(e)
        srv.next_status = 200
        srv.raw_body = json.dumps({"id": "x", "choices": []}).encode("utf-8")
        try:
            call("p")
            raise AssertionError("空 choices 应抛错")
        except RuntimeError as e:
            assert "choices[0].message.content" in str(e)
        srv.raw_body = None
        srv.next_content = "恢复"
        assert call("p") == "恢复", "错误不得污染后续调用"
    _with_server(body)


def test_config_resolution_and_validation(monkeypatch=None):
    import os
    saved = {k: os.environ.pop(k, None) for k in ("OPENAI_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL")}
    try:
        # 缺 model 与 key: 报错点名缺哪些环境变量
        try:
            ad.make_call()
            raise AssertionError("无配置应报错")
        except ValueError as e:
            assert "OPENAI_MODEL" in str(e) and "OPENAI_API_KEY" in str(e), str(e)
        # 环境变量生效, 参数优先
        os.environ["OPENAI_MODEL"] = "env-model"
        os.environ["OPENAI_API_KEY"] = "env-key"
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:1/v1/"
        cfg = ad._config(None, None, None)
        assert cfg == {"model": "env-model", "base_url": "http://127.0.0.1:1/v1", "api_key": "env-key"}
        cfg2 = ad._config("arg-model", None, "arg-key")
        assert cfg2["model"] == "arg-model" and cfg2["api_key"] == "arg-key"
        # 默认 base_url
        del os.environ["OPENAI_BASE_URL"]
        assert ad._config(None, None, None)["base_url"] == "https://api.openai.com/v1"
        # 非法 json_mode
        try:
            ad.make_llm(model="m", api_key="k", json_mode="yaml")
            raise AssertionError("非法 json_mode 应报错")
        except ValueError as e:
            assert "json_mode" in str(e)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


def test_adapter_plugs_into_make_model_and_evaluate():
    """端到端: 适配器产出的 call/llm 能直接喂给 paired_bench.make_model 与 eval_task.evaluate。
    这才是它存在的理由 —— 签名对得上比每个字段都对更重要。"""
    import eval_task as et
    import paired_bench as pb

    def body(srv):
        make = lambda: pb.make_model(ad.make_call(model="m", base_url=srv.base_url, api_key="k"),
                                     sleep=lambda s: None)      # 测试里不真等重试间隔
        srv.next_content = "42"
        assert make()("多少") == "42"
        # 服务端故障: make_model 有界重试后返回 None(该题成对丢弃), 不崩整批
        srv.next_status = 500
        before = len(srv.requests)
        assert make()("多少") is None
        assert len(srv.requests) - before == 2, "tries=2 -> 恰好两次请求"
        srv.next_status = 200
        # judge 路径: 走真的 evaluate(rubric 路由), 假服务扮演评委
        srv.next_content = json.dumps({"verdict": "met", "reasoning": "r"})
        llm = ad.make_llm(model="m", base_url=srv.base_url, api_key="k")
        r = et.evaluate({"id": "t", "instruction": "回答",
                         "verification": {"class": "rubric",
                                          "criteria": [{"text": "给出了数字", "weight": 2},
                                                       {"text": "没有多余解释", "weight": 1}]}},
                        response="42", llm=llm)
        assert r["score"] == 1.0 and r["metrics"]["judged_weight_share"] == 1.0, r["metrics"]
        assert len(r["details"]) == 2 and all(d["verdict"] == "met" for d in r["details"])
        # 评委请求确实带了 schema 约束(评委的结构化输出是 judge 可靠性的一部分)
        assert srv.requests[-1]["body"]["response_format"]["type"] == "json_schema"
    _with_server(body)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
