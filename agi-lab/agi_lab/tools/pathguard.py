"""工作区路径围栏：把模型给出的路径强制解析到工作区根目录内。

这是安全模型的第一道运行时强制：所有文件工具只经由 :class:`PathGuard` 解析目标路径。
``..`` 逃逸、工作区外的绝对路径、指向工作区外的符号链接一律抛
:class:`agi_lab.errors.SafetyViolation`，不做"尽力而为"的兜底。
"""

from __future__ import annotations

import os
from pathlib import Path

from ..errors import SafetyViolation, ToolInvocationError


class PathGuard:
    """把用户路径限定在 ``root`` 目录内的解析器。

    参数：
        root: 工作区根目录；构造时 ``resolve()`` 并保存为绝对路径。
        allow_symlinks: 是否允许路径中出现符号链接。默认 ``False``；此时目标路径自身
            或任一已存在的祖先为符号链接都会被拒绝（即使链接指向工作区内部）。

    异常：
        SafetyViolation: 目标路径逃逸出 ``root``，或路径中出现被禁用的符号链接。
    """

    def __init__(self, root: Path, *, allow_symlinks: bool = False) -> None:
        self._root = Path(root).resolve()
        self._allow_symlinks = allow_symlinks

    @property
    def root(self) -> Path:
        """工作区根目录的绝对路径。"""
        return self._root

    def resolve(self, path_str: str, *, must_exist: bool = False) -> Path:
        """把用户路径解析为 ``root`` 内的绝对路径。

        参数：
            path_str: 相对路径（相对 ``root``），或位于 ``root`` 内的绝对路径。
            must_exist: 为 ``True`` 时要求解析结果已存在。

        返回：
            位于 ``root`` 内、已解析符号链接的绝对路径。

        异常：
            SafetyViolation: 路径逃逸出 ``root``，或路径包含被禁用的符号链接。
            ToolInvocationError: ``must_exist=True`` 而目标不存在，或路径不是字符串。
        """
        if not isinstance(path_str, str):
            raise ToolInvocationError(f"路径必须是字符串，收到 {type(path_str).__name__}")
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        try:
            lexical = Path(os.path.normpath(candidate))
        except (OSError, ValueError) as exc:
            raise SafetyViolation(f"路径无法解析: {path_str}") from exc
        if not lexical.is_relative_to(self._root):
            raise SafetyViolation(f"路径越界，拒绝访问工作区外的路径: {path_str}")
        relative = lexical.relative_to(self._root)
        if not self._allow_symlinks:
            self._reject_symlinks(relative, path_str)
        try:
            resolved = lexical.resolve()
        except (OSError, ValueError) as exc:
            raise SafetyViolation(f"路径无法解析: {path_str}") from exc
        if not resolved.is_relative_to(self._root):
            raise SafetyViolation(f"路径越界（符号链接指向工作区外）: {path_str}")
        if must_exist and not resolved.exists():
            raise ToolInvocationError(f"路径不存在: {path_str}")
        return resolved

    def exists(self, path_str: str) -> bool:
        """目标路径是否存在；越界仍抛 :class:`SafetyViolation`（安全不因查询而放宽）。"""
        return self.resolve(path_str).exists()

    def relative(self, path: Path) -> str:
        """把工作区内的绝对路径转成相对 ``root`` 的 posix 字符串（根目录为 ``"."``）。

        参数：
            path: 绝对路径，或相对 ``root`` 的路径。

        异常：
            SafetyViolation: 路径不在 ``root`` 内。
        """
        target = Path(path)
        if not target.is_absolute():
            target = self._root / target
        try:
            resolved = target.resolve()
            relative = resolved.relative_to(self._root)
        except (OSError, ValueError) as exc:
            raise SafetyViolation(f"路径不在工作区内: {path}") from exc
        return relative.as_posix() if relative.parts else "."

    def _reject_symlinks(self, relative: Path, original: str) -> None:
        current = self._root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise SafetyViolation(f"路径包含符号链接，已禁用: {original}")
