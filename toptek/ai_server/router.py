"""Routing heuristics for selecting LM Studio models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Mapping

from .config import AISettings
from .lmstudio import ModelStats


@dataclass
class ChatTask:
    """Descriptor extracted from a chat request."""

    requires_tools: bool
    expected_tokens: Optional[int]
    reasoning_required: bool
    manual_model: Optional[str] = None


@dataclass
class ModelTelemetry:
    tokens_per_second: Optional[float] = None
    ttft_ms: Optional[float] = None


@dataclass
class RoutingDecision:
    model_id: str
    reason: str


class ModelRouter:
    """Choose the most capable model for the incoming task."""

    def __init__(self, settings: AISettings) -> None:
        self._manual_override: Optional[str] = settings.default_model
        self._models: Dict[str, ModelStats] = {}
        self._telemetry: Dict[str, ModelTelemetry] = {}

    def register_models(self, models: Iterable[ModelStats]) -> None:
        self._models = {model.model_id: model for model in models}
        if self._manual_override and self._manual_override not in self._models:
            self._manual_override = None

    def list_models(self) -> List[ModelStats]:
        return list(self._models.values())

    def manual_override(self, model_id: Optional[str]) -> None:
        if model_id is not None and model_id not in self._models:
            raise KeyError(f"Model {model_id} not available")
        self._manual_override = model_id

    def current_selection(self) -> Optional[str]:
        return self._manual_override

    def record_usage(self, model_id: str, usage: Mapping[str, Any]) -> None:
        telemetry = self._telemetry.setdefault(model_id, ModelTelemetry())
        tokens_per_second = usage.get("tokens_per_second")
        if tokens_per_second is not None:
            try:
                telemetry.tokens_per_second = float(tokens_per_second)
            except (TypeError, ValueError):
                pass
        ttft = usage.get("ttft")
        if ttft is not None:
            try:
                telemetry.ttft_ms = float(ttft)
            except (TypeError, ValueError):
                pass

    def describe(self, model_id: str) -> Dict[str, Optional[float]]:
        telemetry = self._telemetry.get(model_id)
        return {
            "tokens_per_second": telemetry.tokens_per_second if telemetry else None,
            "ttft_ms": telemetry.ttft_ms if telemetry else None,
        }

    def select(self, task: ChatTask) -> RoutingDecision:
        if task.manual_model and task.manual_model in self._models:
            self._manual_override = task.manual_model
            return RoutingDecision(model_id=task.manual_model, reason="manual-request")

        if self._manual_override and self._manual_override in self._models:
            return RoutingDecision(
                model_id=self._manual_override, reason="manual-override"
            )

        if not self._models:
            raise RuntimeError("No LM Studio models available")

        candidates = list(self._models.values())

        if task.requires_tools:
            tool_ready = [model for model in candidates if model.supports_tools]
            if tool_ready:
                candidates = tool_ready

        if task.expected_tokens is not None:
            sized = [
                model
                for model in candidates
                if model.max_context is None
                or model.max_context >= task.expected_tokens
            ]
            if sized:
                candidates = sized

        if task.reasoning_required:
            reasoning = [model for model in candidates if model.is_reasoning_model]
            if reasoning:
                candidates = reasoning

        # Prefer the model with the highest observed throughput, falling back to context window.
        def model_score(model: ModelStats) -> float:
            telemetry = self._telemetry.get(model.model_id)
            if telemetry and telemetry.tokens_per_second:
                return telemetry.tokens_per_second
            if model.tokens_per_second:
                return model.tokens_per_second
            return float(model.max_context or 0)

        selected = max(candidates, key=model_score)
        return RoutingDecision(model_id=selected.model_id, reason="auto-policy")


def infer_task(
    *,
    tools: Optional[List[dict]],
    max_tokens: Optional[int],
    system_role: Optional[str],
    manual_model: Optional[str] = None,
) -> ChatTask:
    requires_tools = bool(tools)
    reasoning_required = False
    if system_role:
        reasoning_required = (
            "reason" in system_role.lower() or "planner" in system_role.lower()
        )
    return ChatTask(
        requires_tools=requires_tools,
        expected_tokens=max_tokens,
        reasoning_required=reasoning_required,
        manual_model=manual_model,
    )


__all__ = ["ChatTask", "ModelRouter", "RoutingDecision", "infer_task"]
