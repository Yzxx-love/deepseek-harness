"""OpenAI 兼容 ``/chat/completions`` provider（DeepSeek / OpenAI）。

请求头、payload 与响应解析遵循 OpenAI Chat Completions 协议。HTTP 传输默认用
标准库 ``urllib.request``，可注入 ``http_post`` 以便离线测试；重试由
:class:`agi_lab.llm.base.BaseLLM` 统一处理。

安全约定：``api_key`` 只出现在 Authorization 请求头，任何异常消息与格式错误摘要
都经过脱敏，不会泄漏密钥。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable, Mapping, Sequence

from ..errors import ConfigError, LLMResponseFormatError, LLMTransportError
from ..types import JSONObject, LLMResponse, Message, Usage
from .base import BaseLLM, RetryPolicy

CHAT_COMPLETIONS_PATH = "/chat/completions"

HttpPostFn = Callable[[str, dict[str, str], JSONObject, float], tuple[int, bytes]]
"""可注入的 HTTP POST：``(url, headers, payload, timeout) -> (status, body_bytes)``。

4xx/5xx 应以返回值交给 provider 分类；网络错误与超时抛 :class:`LLMTransportError`
或 ``urllib.error.URLError`` / ``TimeoutError``。
"""


class OpenAICompatibleLLM(BaseLLM):
    """面向 OpenAI 兼容 ``/chat/completions`` 的 :class:`agi_lab.types.LLMProvider` 实现。

    参数：
        model: 模型名，如 ``deepseek-chat``。
        api_key: 提供方密钥，必填且非空。
        base_url: 提供方根地址，可带或不带尾部 ``/``，也可已含 ``/chat/completions``。
        timeout: 默认请求超时秒数，``complete(timeout=...)`` 可覆盖。
        default_temperature: ``complete(temperature=None)`` 时使用的采样温度。
        default_max_tokens: ``complete(max_tokens=None)`` 时使用的输出上限。
        name: provider 名称，默认 ``deepseek``。
        http_post: HTTP 传输注入点；``None`` 用标准库实现。
        retry: 重试策略，透传 :class:`agi_lab.llm.base.BaseLLM`。
        sleep: 等待函数，透传 :class:`agi_lab.llm.base.BaseLLM`。
        jitter: 乘性抖动因子，透传 :class:`agi_lab.llm.base.BaseLLM`。

    属性：
        endpoint: 规范化后的完整 ``/chat/completions`` URL。
        model: 模型名。
        base_url: 构造时传入的根地址。
        timeout: 默认请求超时秒数。
        default_temperature: 默认采样温度。
        default_max_tokens: 默认输出上限。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 120.0,
        default_temperature: float = 0.2,
        default_max_tokens: int = 4096,
        name: str = "deepseek",
        http_post: HttpPostFn | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        """构造 provider。

        异常：
            ConfigError: ``api_key`` 为空或 ``base_url`` 为空。
        """
        if not api_key:
            raise ConfigError("provider=deepseek 需要 DEEPSEEK_API_KEY（或 --api-key）")
        super().__init__(name=name, retry=retry, sleep=sleep, jitter=jitter)
        self.model = model
        self.base_url = base_url
        self.endpoint = _chat_completions_url(base_url)
        self.timeout = timeout
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._api_key = api_key
        self._http_post: HttpPostFn = http_post if http_post is not None else _urllib_post

    def _complete_once(
        self,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        timeout: float | None,
    ) -> LLMResponse:
        """发送一次 ``/chat/completions`` 请求并解析响应，不含重试。

        异常：
            LLMTransportError: 网络/超时失败（``retryable=True``），或 HTTP 非 2xx
                （状态码决定 ``retryable``）。
            LLMResponseFormatError: 响应体非 JSON、缺 ``choices`` 或内容为空。
        """
        payload = self._build_payload(messages, temperature, max_tokens, json_mode)
        request_timeout = self.timeout if timeout is None else timeout
        try:
            status, body = self._http_post(
                self.endpoint, self._headers(), payload, request_timeout
            )
        except LLMTransportError:
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMTransportError(
                self._sanitize(f"请求 {self.endpoint} 失败：{exc}"), retryable=True
            ) from exc
        if not 200 <= status < 300:
            raise self._status_error(status, body)
        return self._parse_response(body)

    def _build_payload(
        self,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> JSONObject:
        payload: JSONObject = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _status_error(self, status: int, body: bytes) -> LLMTransportError:
        retryable = status in self.retry.retry_statuses or 500 <= status <= 599
        excerpt = self._excerpt(_decode_body(body))
        message = f"HTTP {status}" + (f"：{excerpt}" if excerpt else "")
        return LLMTransportError(self._sanitize(message), status=status, retryable=retryable)

    def _parse_response(self, body: bytes) -> LLMResponse:
        text = _decode_body(body)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError(
                "模型响应不是合法 JSON", excerpt=self._excerpt(text)
            ) from exc
        if not isinstance(data, Mapping):
            raise LLMResponseFormatError(
                "模型响应顶层不是 JSON 对象", excerpt=self._excerpt(text)
            )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseFormatError(
                "模型响应缺少 choices", excerpt=self._excerpt(text)
            )
        first = choices[0]
        if not isinstance(first, Mapping):
            raise LLMResponseFormatError(
                "模型响应 choices[0] 不是对象", excerpt=self._excerpt(text)
            )
        content = _message_content(first.get("message"))
        if not content.strip():
            raise LLMResponseFormatError(
                "模型响应内容为空", excerpt=self._excerpt(text)
            )
        finish_reason = first.get("finish_reason")
        finish = finish_reason if isinstance(finish_reason, str) else ""
        model = data.get("model")
        usage = _parse_usage(data.get("usage"))
        raw: JSONObject = {
            "finish_reason": finish,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
        }
        return LLMResponse(
            text=content,
            model=model if isinstance(model, str) else "",
            usage=usage,
            finish_reason=finish,
            raw=raw,
        )

    def _sanitize(self, text: str) -> str:
        """把消息中出现的 api_key 替换为 ``***``。"""
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "***")
        return text

    def _excerpt(self, text: str, limit: int = 300) -> str:
        """先脱敏再截断，避免截断切口留下部分密钥。"""
        sanitized = self._sanitize(text).strip()
        return sanitized if len(sanitized) <= limit else sanitized[:limit] + "…"


def _chat_completions_url(base_url: str) -> str:
    """把 ``base_url`` 规范化为完整 ``/chat/completions`` 地址。

    参数：
        base_url: 提供方根地址；尾部 ``/`` 会被去掉，已含 ``/chat/completions`` 时原样使用。

    返回：
        完整请求 URL。

    异常：
        ConfigError: ``base_url`` 为空。
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise ConfigError("base_url 不能为空")
    if url.endswith(CHAT_COMPLETIONS_PATH):
        return url
    return url + CHAT_COMPLETIONS_PATH


def _message_content(message: object) -> str:
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _parse_usage(usage: object) -> Usage:
    if not isinstance(usage, Mapping):
        return Usage()
    return Usage(
        prompt_tokens=_as_token_count(usage.get("prompt_tokens")),
        completion_tokens=_as_token_count(usage.get("completion_tokens")),
    )


def _as_token_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


def _decode_body(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _urllib_post(
    url: str, headers: dict[str, str], payload: JSONObject, timeout: float
) -> tuple[int, bytes]:
    """默认 HTTP 传输：用标准库 ``urllib.request`` 发送 JSON POST。

    参数：
        url: 完整请求 URL。
        headers: 请求头；Authorization 只在此处出现。
        payload: 请求体 JSON 对象。
        timeout: 超时秒数。

    返回：
        ``(HTTP 状态码, 响应体字节)``；4xx/5xx 以返回值表达，由 provider 分类。

    异常：
        LLMTransportError: 连接失败、DNS 失败或超时，``retryable=True``。
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        try:
            return int(exc.code), exc.read()
        finally:
            exc.close()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMTransportError(f"请求 {url} 失败：{exc}", retryable=True) from exc
