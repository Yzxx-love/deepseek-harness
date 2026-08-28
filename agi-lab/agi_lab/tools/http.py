"""http_get 工具：只允许 http/https 的只读抓取。

默认不注册，必须由 ``build_default_tools(allow_network=True)`` 显式开启；且
``requires_approval=True``，由注册表审批。固定 User-Agent，20 秒超时，响应体截断。
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

from ..errors import ToolInvocationError
from ..types import JSONObject, ToolResult, ToolSpec
from .base import DEFAULT_TRUNCATE_CHARS, BaseTool, truncate_text

USER_AGENT = "agi-lab/0.1"
REQUEST_TIMEOUT_SECONDS = 20.0
MAX_READ_BYTES = 256_000
ALLOWED_SCHEMES = ("http", "https")


class HttpFetchTool(BaseTool):
    """抓取一个 http/https URL 的响应正文。"""

    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="http_get",
            description="抓取 http/https URL 的响应正文（只读，需审批）",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "完整的 http 或 https URL"},
                },
                "required": ["url"],
            },
            requires_approval=True,
        )

    def _execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        url = self._require_str(arguments, "url")
        parsed = urllib.parse.urlsplit(url)
        scheme = parsed.scheme.lower()
        if scheme not in ALLOWED_SCHEMES:
            raise ToolInvocationError(f"只允许 http/https URL，收到 scheme={scheme or '(空)'}")
        if not parsed.netloc:
            raise ToolInvocationError(f"URL 缺少主机名: {url}")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = int(getattr(response, "status", 0) or 0)
                charset = response.headers.get_content_charset() or "utf-8"
                content_type = response.headers.get_content_type()
                raw = response.read(MAX_READ_BYTES)
        except urllib.error.HTTPError as exc:
            return ToolResult(
                ok=False,
                error=f"HTTP {exc.code}: {exc.reason}",
                metadata={"url": url, "status": int(exc.code)},
            )
        except urllib.error.URLError as exc:
            return ToolResult(ok=False, error=f"网络请求失败: {exc.reason}", metadata={"url": url})
        except TimeoutError:
            return ToolResult(
                ok=False,
                error=f"请求超时（{REQUEST_TIMEOUT_SECONDS:g} 秒）",
                metadata={"url": url, "timed_out": True},
            )
        except OSError as exc:
            return ToolResult(ok=False, error=f"网络请求失败: {exc}", metadata={"url": url})
        try:
            text = raw.decode(charset, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
        metadata: JSONObject = {
            "url": url,
            "status": status,
            "content_type": content_type,
            "bytes": len(raw),
            "truncated": len(text) > DEFAULT_TRUNCATE_CHARS,
        }
        return ToolResult(ok=True, output=truncate_text(text, DEFAULT_TRUNCATE_CHARS), metadata=metadata)
