"""Pytest hook extensions that enforce >=90% line coverage using ``trace``."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import threading
import trace
import json
import types
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest

_SKIP_TRACE_COVERAGE = bool(os.environ.get("TOPTEK_SKIP_TRACE_COVERAGE"))

_TARGET_MODULES = (
    "toptek.core.data",
    "toptek.loops.learn",
    "toptek.model.threshold",
    "toptek.core.utils",
    "toptek.core.risk",
    "toptek.gui.app",
)

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")

    def _parse_scalar(value: str) -> object:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _safe_load(data: str | bytes | None) -> dict:
        if data is None:
            return {}
        if hasattr(data, "read"):
            text = data.read()
        elif isinstance(data, bytes):
            text = data.decode("utf-8")
        else:
            text = str(data)
        root: dict[str, object] = {}
        stack: list[tuple[int, dict[str, object]]] = [(0, root)]
        for raw_line in text.splitlines():
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            while stack and indent < stack[-1][0]:
                stack.pop()
            current = stack[-1][1]
            key, _, value_part = raw_line.strip().partition(":")
            if not _:
                continue
            value_part = value_part.strip()
            if not value_part:
                nested: dict[str, object] = {}
                current[key] = nested
                stack.append((indent + 2, nested))
            else:
                current[key] = _parse_scalar(value_part)
        return root

    def _safe_dump(data: object, stream=None, **_kwargs) -> str | None:
        payload = json.dumps(data)
        if stream is None:
            return payload
        stream.write(payload)
        return None

    yaml_stub.safe_load = _safe_load  # type: ignore[attr-defined]
    yaml_stub.safe_dump = _safe_dump  # type: ignore[attr-defined]
    sys.modules["yaml"] = yaml_stub


def _code_lines(path: Path) -> set[int]:
    lines: set[int] = set()
    in_docstring = False
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                else:
                    if (
                        not (stripped.endswith('"""') or stripped.endswith("'''"))
                        or len(stripped) == 3
                    ):
                        in_docstring = True
                continue
            if in_docstring:
                if stripped.endswith('"""') or stripped.endswith("'''"):
                    in_docstring = False
                continue
            lines.add(number)
    return lines


def _module_filename(module_name: str) -> Path | None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    filename = inspect.getsourcefile(module)
    if not filename:
        raise RuntimeError(f"Unable to resolve source file for {module_name}")
    return Path(filename).resolve()


def _coverage_for_module(
    results: trace.CoverageResults, module_path: Path
) -> Tuple[float, int, int, set[int]]:
    executed = {
        lineno
        for (filename, lineno), count in results.counts.items()
        if Path(filename).resolve() == module_path and count > 0
    }
    code_lines = _code_lines(module_path)
    if not code_lines:
        return 1.0, 0, 0, set()
    covered = len(executed & code_lines)
    total = len(code_lines)
    missing = code_lines - executed
    return covered / total if total else 1.0, covered, total, missing


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    if _SKIP_TRACE_COVERAGE:
        return
    tracer = trace.Trace(
        count=True, trace=False, ignoremods=("pytest", "_pytest", "pluggy")
    )
    config = session.config
    config._toptek_tracer = tracer  # type: ignore[attr-defined]
    config._toptek_prev_trace = sys.gettrace()  # type: ignore[attr-defined]
    config._toptek_prev_thread_trace = threading.gettrace()  # type: ignore[attr-defined]
    sys.settrace(tracer.globaltrace)
    threading.settrace(tracer.globaltrace)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _SKIP_TRACE_COVERAGE:
        return
    config = session.config
    tracer: trace.Trace | None = getattr(config, "_toptek_tracer", None)
    if tracer is None:
        return
    sys.settrace(getattr(config, "_toptek_prev_trace", None))
    threading.settrace(getattr(config, "_toptek_prev_thread_trace", None))
    results = tracer.results()

    summary: List[Tuple[str, float, int, int]] = []
    failures: List[Tuple[str, float]] = []
    for module_name in _TARGET_MODULES:
        if module_name == "toptek.gui.app" and not os.environ.get("DISPLAY"):
            summary.append((module_name, 0.0, 0, 0))
            continue
        module_path = _module_filename(module_name)
        if module_path is None:
            summary.append((module_name, 0.0, 0, 0))
            continue
        ratio, covered, total, missing = _coverage_for_module(results, module_path)
        summary.append((module_name, ratio, covered, total))
        if ratio < 0.90:
            failures.append((module_name, ratio, sorted(missing)))

    config._toptek_coverage_summary = summary  # type: ignore[attr-defined]
    if failures:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        config._toptek_coverage_failures = failures  # type: ignore[attr-defined]


def pytest_terminal_summary(
    terminalreporter, exitstatus: int, config: pytest.Config
) -> None:
    if _SKIP_TRACE_COVERAGE:
        return
    summary: Iterable[Tuple[str, float, int, int]] = getattr(
        config, "_toptek_coverage_summary", []
    )
    if not summary:
        return
    terminalreporter.write_sep("-", "Module coverage (trace)")
    for module_name, ratio, covered, total in summary:
        if total == 0:
            terminalreporter.write_line(
                f"{module_name:<30} skipped (dependency unavailable)"
            )
            continue
        terminalreporter.write_line(
            f"{module_name:<30} {covered}/{total} lines -> {ratio:.1%}"
        )
    failures: Iterable[Tuple[str, float, Iterable[int]]] = getattr(
        config, "_toptek_coverage_failures", []
    )
    if failures:
        failing = ", ".join(
            f"{name}={ratio:.1%} (missing {list(missing)[:5]})"
            for name, ratio, missing in failures
        )
        terminalreporter.write_line(
            f"Coverage threshold not met (>=90% required): {failing}",
            red=True,
        )
