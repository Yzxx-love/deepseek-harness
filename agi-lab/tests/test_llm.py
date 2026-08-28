"""agi_lab.llm 的单元测试。

全部测试离线执行：HTTP 由 ``FakeHttp`` 注入，等待由 ``sleep`` 注入，不触碰网络、
不产生真实退避。断言覆盖成功解析、payload/headers 组装、状态码分类、重试与退避、
格式错误不重试、密钥脱敏、URL 拼接与配置驱动工厂。
"""

from __future__ import annotations

import json
import unittest
import urllib.error
from typing import Any, Sequence

from agi_lab.config import Config
from agi_lab.errors import ConfigError, LLMResponseFormatError, LLMTransportError
from agi_lab.llm import BaseLLM, OpenAICompatibleLLM, RetryPolicy, build_llm
from agi_lab.types import JSONObject, LLMProvider, LLMResponse, Message, Usage

API_KEY = "sk-unit-test-secret"


def _response_body(
    text: str = "你好",
    *,
    model: str = "deepseek-chat",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
    include_usage: bool = True,
) -> bytes:
    """构造一份 OpenAI 兼容的成功响应体。"""
    payload: dict[str, Any] = {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }
    if include_usage:
        payload["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class FakeHttp:
    """记录调用参数并按脚本返回的 HTTP 桩。

    参数：
        responses: 依次返回的 ``(status, body)``；元素也可以异常实例，表示传输层失败。
            调用次数超过脚本长度即抛 ``AssertionError``，让"不该发生的重试"直接暴露。
    """

    def __init__(self, responses: list[tuple[int, bytes] | BaseException]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, str], JSONObject, float]] = []

    def __call__(
        self, url: str, headers: dict[str, str], payload: JSONObject, timeout: float
    ) -> tuple[int, bytes]:
        self.calls.append((url, dict(headers), dict(payload), timeout))
        if len(self.calls) > len(self._responses):
            raise AssertionError(f"意外的第 {len(self.calls)} 次 HTTP 调用")
        response = self._responses[len(self.calls) - 1]
        if isinstance(response, BaseException):
            raise response
        return response


class ProviderTestBase(unittest.TestCase):
    """构造带 HTTP 桩与 sleep 记录的 provider。"""

    def make_provider(
        self,
        responses: list[tuple[int, bytes] | BaseException],
        *,
        retry: RetryPolicy | None = None,
        api_key: str = API_KEY,
        base_url: str = "https://api.deepseek.com",
        **overrides: Any,
    ) -> tuple[OpenAICompatibleLLM, FakeHttp, list[float]]:
        fake = FakeHttp(responses)
        sleeps: list[float] = []
        provider = OpenAICompatibleLLM(
            model="deepseek-chat",
            api_key=api_key,
            base_url=base_url,
            http_post=fake,
            retry=retry,
            sleep=sleeps.append,
            **overrides,
        )
        return provider, fake, sleeps


class ParsingTest(ProviderTestBase):
    """成功响应解析与用量统计。"""

    def test_parses_text_model_usage_and_finish_reason(self) -> None:
        provider, fake, sleeps = self.make_provider([(200, _response_body("你好，世界"))])
        response = provider.complete([Message(role="user", content="hi")])

        self.assertEqual(response.text, "你好，世界")
        self.assertEqual(response.model, "deepseek-chat")
        self.assertEqual(response.usage, Usage(prompt_tokens=10, completion_tokens=5))
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(
            response.raw,
            {
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
        self.assertNotIn("choices", response.raw or {})
        self.assertEqual(sleeps, [])
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(provider.total_usage, Usage(10, 5))

    def test_missing_usage_defaults_to_zero(self) -> None:
        provider, _, _ = self.make_provider([(200, _response_body(include_usage=False))])
        response = provider.complete([Message(role="user", content="hi")])

        self.assertEqual(response.usage, Usage())
        self.assertEqual(response.usage.total_tokens, 0)

    def test_total_usage_sums_across_calls(self) -> None:
        provider, fake, _ = self.make_provider(
            [
                (200, _response_body("一", prompt_tokens=10, completion_tokens=5)),
                (200, _response_body("二", prompt_tokens=3, completion_tokens=4)),
            ]
        )
        provider.complete([Message(role="user", content="first")])
        provider.complete([Message(role="user", content="second")])

        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(provider.total_usage, Usage(13, 9))

    def test_implements_llm_provider_protocol(self) -> None:
        provider, _, _ = self.make_provider([(200, _response_body())])
        self.assertIsInstance(provider, LLMProvider)
        self.assertEqual(provider.name, "deepseek")

    def test_non_json_body_raises_format_error_without_retry(self) -> None:
        provider, fake, sleeps = self.make_provider([(200, b"<html>not json</html>")])

        with self.assertRaises(LLMResponseFormatError):
            provider.complete([Message(role="user", content="hi")])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(sleeps, [])

    def test_malformed_response_bodies_raise_format_error(self) -> None:
        malformed = [
            json.dumps({"model": "m", "choices": []}).encode("utf-8"),
            json.dumps({"model": "m"}).encode("utf-8"),
            json.dumps({"choices": [{"message": {"content": ""}}]}).encode("utf-8"),
            json.dumps({"choices": [{"message": {"content": "   "}}]}).encode("utf-8"),
            json.dumps({"choices": [{"message": None}]}).encode("utf-8"),
            json.dumps([1, 2, 3]).encode("utf-8"),
        ]
        for body in malformed:
            with self.subTest(body=body):
                provider, fake, _ = self.make_provider([(200, body)])
                with self.assertRaises(LLMResponseFormatError):
                    provider.complete([Message(role="user", content="hi")])
                self.assertEqual(len(fake.calls), 1)


class RequestBuildingTest(ProviderTestBase):
    """payload、请求头、消息序列化与 URL 拼接。"""

    def test_default_payload(self) -> None:
        provider, fake, _ = self.make_provider([(200, _response_body())])
        provider.complete([Message(role="user", content="hi")])

        _, headers, payload, timeout = fake.calls[0]
        self.assertEqual(
            payload,
            {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.2,
                "max_tokens": 4096,
                "stream": False,
            },
        )
        self.assertNotIn("response_format", payload)
        self.assertEqual(timeout, 120.0)
        self.assertEqual(
            headers,
            {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    def test_json_mode_and_parameter_overrides(self) -> None:
        provider, fake, _ = self.make_provider([(200, _response_body())])
        provider.complete(
            [Message(role="user", content="hi")],
            temperature=0.7,
            max_tokens=128,
            json_mode=True,
        )

        _, _, payload, _ = fake.calls[0]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 128)

    def test_messages_are_serialized_through_to_dict(self) -> None:
        messages = [
            Message(role="system", content="只输出 JSON"),
            Message(role="user", content="hi", name="alice"),
        ]
        provider, fake, _ = self.make_provider([(200, _response_body())])
        provider.complete(messages)

        _, _, payload, _ = fake.calls[0]
        self.assertEqual(payload["messages"], [message.to_dict() for message in messages])
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "只输出 JSON"},
                {"role": "user", "content": "hi", "name": "alice"},
            ],
        )

    def test_timeout_override_is_passed_to_http(self) -> None:
        provider, fake, _ = self.make_provider(
            [(200, _response_body()), (200, _response_body())]
        )
        provider.complete([Message(role="user", content="hi")], timeout=9.5)
        provider.complete([Message(role="user", content="hi")])

        self.assertEqual([call[3] for call in fake.calls], [9.5, 120.0])

    def test_endpoint_url_joining(self) -> None:
        cases = [
            ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
            ("https://api.deepseek.com/", "https://api.deepseek.com/chat/completions"),
            ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/chat/completions"),
            ("https://api.deepseek.com/v1/", "https://api.deepseek.com/v1/chat/completions"),
            (
                "https://example.test/openai/chat/completions",
                "https://example.test/openai/chat/completions",
            ),
            (
                "https://example.test/openai/chat/completions/",
                "https://example.test/openai/chat/completions",
            ),
        ]
        for base_url, expected in cases:
            with self.subTest(base_url=base_url):
                provider, fake, _ = self.make_provider([(200, _response_body())], base_url=base_url)
                provider.complete([Message(role="user", content="hi")])
                self.assertEqual(provider.endpoint, expected)
                self.assertEqual(fake.calls[0][0], expected)

    def test_api_key_is_required(self) -> None:
        with self.assertRaises(ConfigError):
            OpenAICompatibleLLM(model="deepseek-chat", api_key="")

    def test_empty_base_url_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            OpenAICompatibleLLM(model="deepseek-chat", api_key=API_KEY, base_url="/")


class RetryTest(ProviderTestBase):
    """状态码分类、重试次数与退避等待。"""

    def test_429_then_success(self) -> None:
        provider, fake, sleeps = self.make_provider(
            [(429, b'{"error":"rate limited"}'), (200, _response_body("恢复"))]
        )
        response = provider.complete([Message(role="user", content="hi")])

        self.assertEqual(response.text, "恢复")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(sleeps, [0.5])
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(provider.total_usage, Usage(10, 5))

    def test_503_exhausts_attempts_with_retryable_transport_error(self) -> None:
        provider, fake, sleeps = self.make_provider([(503, b'{"error":"unavailable"}')] * 3)

        with self.assertRaises(LLMTransportError) as caught:
            provider.complete([Message(role="user", content="hi")])

        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(sleeps, [0.5, 1.0])
        self.assertEqual(caught.exception.status, 503)
        self.assertIs(caught.exception.retryable, True)
        self.assertEqual(provider.call_count, 0)
        self.assertEqual(provider.total_usage, Usage())

    def test_non_retryable_status_calls_once(self) -> None:
        provider, fake, sleeps = self.make_provider([(401, b'{"error":"invalid key"}')])

        with self.assertRaises(LLMTransportError) as caught:
            provider.complete([Message(role="user", content="hi")])

        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(caught.exception.status, 401)
        self.assertIs(caught.exception.retryable, False)

    def test_status_classification(self) -> None:
        cases = [
            (400, False),
            (401, False),
            (403, False),
            (404, False),
            (422, False),
            (408, True),
            (409, True),
            (429, True),
            (500, True),
            (502, True),
            (503, True),
            (504, True),
            (599, True),
        ]
        for status, retryable in cases:
            with self.subTest(status=status):
                provider, fake, sleeps = self.make_provider(
                    [(status, b'{"error":"x"}')],
                    retry=RetryPolicy(max_attempts=1),
                )
                with self.assertRaises(LLMTransportError) as caught:
                    provider.complete([Message(role="user", content="hi")])
                self.assertEqual(len(fake.calls), 1)
                self.assertEqual(sleeps, [])
                self.assertEqual(caught.exception.status, status)
                self.assertIs(caught.exception.retryable, retryable)

    def test_exponential_backoff_is_capped_by_max_delay(self) -> None:
        retry = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=2.5)
        provider, fake, sleeps = self.make_provider([(503, b"down")] * 5, retry=retry)

        with self.assertRaises(LLMTransportError):
            provider.complete([Message(role="user", content="hi")])

        self.assertEqual(len(fake.calls), 5)
        self.assertEqual(sleeps, [1.0, 2.0, 2.5, 2.5])

    def test_timeout_is_retryable_and_retried(self) -> None:
        provider, fake, sleeps = self.make_provider(
            [TimeoutError("timed out"), (200, _response_body("恢复"))]
        )
        response = provider.complete([Message(role="user", content="hi")])

        self.assertEqual(response.text, "恢复")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(sleeps, [0.5])

    def test_url_error_exhausts_as_retryable_transport_error(self) -> None:
        provider, fake, sleeps = self.make_provider(
            [urllib.error.URLError("dns failed")] * 2,
            retry=RetryPolicy(max_attempts=2),
        )

        with self.assertRaises(LLMTransportError) as caught:
            provider.complete([Message(role="user", content="hi")])

        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(sleeps, [0.5])
        self.assertIs(caught.exception.retryable, True)
        self.assertIn("dns failed", str(caught.exception))

    def test_format_error_is_not_retried_even_when_retries_remain(self) -> None:
        provider, fake, sleeps = self.make_provider([(200, b"not-json")])

        with self.assertRaises(LLMResponseFormatError):
            provider.complete([Message(role="user", content="hi")])

        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(sleeps, [])

    def test_error_messages_do_not_leak_api_key(self) -> None:
        provider, _, _ = self.make_provider(
            [(500, b'upstream rejected token sk-unit-test-secret')],
            retry=RetryPolicy(max_attempts=1),
        )
        with self.assertRaises(LLMTransportError) as caught:
            provider.complete([Message(role="user", content="hi")])
        self.assertNotIn(API_KEY, str(caught.exception))
        self.assertIn("***", str(caught.exception))

        provider, _, _ = self.make_provider(
            [(200, b'{"broken": "sk-unit-test-secret"')]
        )
        with self.assertRaises(LLMResponseFormatError) as caught_format:
            provider.complete([Message(role="user", content="hi")])
        self.assertNotIn(API_KEY, str(caught_format.exception))
        self.assertNotIn(API_KEY, caught_format.exception.excerpt)

    def test_key_is_sanitized_before_truncation(self) -> None:
        body = b"x" * 290 + API_KEY.encode("utf-8")
        provider, _, _ = self.make_provider(
            [(500, body)], retry=RetryPolicy(max_attempts=1)
        )
        with self.assertRaises(LLMTransportError) as caught:
            provider.complete([Message(role="user", content="hi")])
        message = str(caught.exception)
        self.assertNotIn(API_KEY, message)
        self.assertNotIn(API_KEY[:10], message)
        self.assertIn("***", message)


class ScriptedLLM(BaseLLM):
    """按脚本依次返回响应或抛异常的 :class:`BaseLLM` 桩。"""

    def __init__(
        self, outcomes: list[LLMResponse | BaseException], **kwargs: Any
    ) -> None:
        super().__init__(name="scripted", **kwargs)
        self._outcomes = list(outcomes)
        self.attempts = 0

    def _complete_once(
        self,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        timeout: float | None,
    ) -> LLMResponse:
        self.attempts += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RetryPolicyTest(unittest.TestCase):
    """RetryPolicy 的默认值与退避公式。"""

    def test_defaults(self) -> None:
        policy = RetryPolicy()
        self.assertEqual(policy.max_attempts, 3)
        self.assertEqual(policy.base_delay, 0.5)
        self.assertEqual(policy.max_delay, 8.0)
        self.assertEqual(policy.retry_statuses, (408, 409, 429, 500, 502, 503, 504))

    def test_delay_for_doubles_and_caps(self) -> None:
        policy = RetryPolicy(max_attempts=6, base_delay=0.5, max_delay=8.0)
        self.assertEqual(
            [policy.delay_for(attempt) for attempt in range(1, 7)],
            [0.5, 1.0, 2.0, 4.0, 8.0, 8.0],
        )

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RetryPolicy(max_attempts=0)
        with self.assertRaises(ValueError):
            RetryPolicy(base_delay=-1.0)
        with self.assertRaises(ValueError):
            RetryPolicy().delay_for(0)


class BaseLLMTest(unittest.TestCase):
    """BaseLLM 模板方法的重试与统计语义。"""

    def test_retryable_error_is_retried_with_backoff(self) -> None:
        sleeps: list[float] = []
        llm = ScriptedLLM(
            [
                LLMTransportError("boom", retryable=True),
                LLMTransportError("boom", retryable=True),
                LLMResponse(text="ok", usage=Usage(2, 3)),
            ],
            retry=RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=4.0),
            sleep=sleeps.append,
        )
        response = llm.complete([Message(role="user", content="hi")])

        self.assertEqual(response.text, "ok")
        self.assertEqual(llm.attempts, 3)
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(llm.total_usage, Usage(2, 3))

    def test_non_retryable_error_stops_immediately(self) -> None:
        sleeps: list[float] = []
        llm = ScriptedLLM(
            [LLMTransportError("fatal", status=401, retryable=False)],
            sleep=sleeps.append,
        )
        with self.assertRaises(LLMTransportError):
            llm.complete([Message(role="user", content="hi")])

        self.assertEqual(llm.attempts, 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(llm.call_count, 0)

    def test_format_error_is_propagated_without_retry(self) -> None:
        sleeps: list[float] = []
        llm = ScriptedLLM(
            [LLMResponseFormatError("bad payload")],
            sleep=sleeps.append,
        )
        with self.assertRaises(LLMResponseFormatError):
            llm.complete([Message(role="user", content="hi")])

        self.assertEqual(llm.attempts, 1)
        self.assertEqual(sleeps, [])

    def test_jitter_multiplies_backoff_delay(self) -> None:
        sleeps: list[float] = []
        llm = ScriptedLLM(
            [
                LLMTransportError("boom", retryable=True),
                LLMResponse(text="ok", usage=Usage(1, 1)),
            ],
            retry=RetryPolicy(max_attempts=3, base_delay=2.0, max_delay=8.0),
            sleep=sleeps.append,
            jitter=lambda: 0.5,
        )
        llm.complete([Message(role="user", content="hi")])

        self.assertEqual(sleeps, [1.0])

    def test_usage_accumulates_only_for_successful_calls(self) -> None:
        sleeps: list[float] = []
        llm = ScriptedLLM(
            [
                LLMTransportError("boom", retryable=True),
                LLMResponse(text="one", usage=Usage(1, 2)),
                LLMResponse(text="two", usage=Usage(3, 4)),
            ],
            retry=RetryPolicy(max_attempts=3, base_delay=0.0),
            sleep=sleeps.append,
        )
        llm.complete([Message(role="user", content="first")])
        llm.complete([Message(role="user", content="second")])

        self.assertEqual(llm.call_count, 2)
        self.assertEqual(llm.total_usage, Usage(4, 6))


class BuildLLMTest(unittest.TestCase):
    """build_llm 的 provider 选择与配置透传。"""

    def test_build_deepseek_from_config(self) -> None:
        config = Config(
            provider="deepseek",
            api_key="sk-from-config",
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com/v1",
            request_timeout=30.0,
            temperature=0.5,
            max_output_tokens=1024,
        )
        provider = build_llm(config)

        self.assertIsInstance(provider, OpenAICompatibleLLM)
        self.assertEqual(provider.name, "deepseek")
        self.assertEqual(provider.model, "deepseek-reasoner")
        self.assertEqual(provider.endpoint, "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(provider.timeout, 30.0)
        self.assertEqual(provider.default_temperature, 0.5)
        self.assertEqual(provider.default_max_tokens, 1024)

    def test_build_mock_returns_mock_llm(self) -> None:
        try:
            from agi_lab.llm.mock import MockLLM
        except ImportError:
            self.skipTest("agi_lab.llm.mock 尚未提供（由 Lead 负责），稍后重跑")

        provider = build_llm(Config(provider="mock"))
        self.assertIsInstance(provider, MockLLM)
        self.assertEqual(provider.name, "mock")

    def test_build_missing_key_raises_config_error(self) -> None:
        class _KeylessConfig:
            # 绕过 Config 构造期校验，专门覆盖工厂的兜底分支
            provider = "deepseek"
            model = "deepseek-chat"
            api_key = None
            base_url = "https://api.deepseek.com"
            request_timeout = 120.0
            temperature = 0.2
            max_output_tokens = 4096

        with self.assertRaises(ConfigError):
            build_llm(_KeylessConfig())  # type: ignore[arg-type]

    def test_build_rejects_unknown_provider(self) -> None:
        class _UnknownProviderConfig:
            provider = "bogus"
            model = "m"
            api_key = None
            base_url = "https://api.deepseek.com"
            request_timeout = 120.0
            temperature = 0.2
            max_output_tokens = 4096

        with self.assertRaises(ConfigError):
            build_llm(_UnknownProviderConfig())  # type: ignore[arg-type]

    def test_config_itself_rejects_deepseek_without_key(self) -> None:
        with self.assertRaises(ConfigError):
            Config(provider="deepseek")


if __name__ == "__main__":
    unittest.main()
