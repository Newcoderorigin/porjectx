"""Guard policy loader and evaluation helpers for Toptek risk."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

try:
    import yaml  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "PyYAML is required to load guard policies. Install it with 'pip install pyyaml'."
    ) from exc

from toptek.core import risk as core_risk

_POLICY_FILENAME = "policy.yml"
_GUARD_OK = "OK"
_GUARD_DEFENSIVE = "DEFENSIVE_MODE"


@dataclass(frozen=True)
class GuardRuleResult:
    """Outcome for a single guard policy rule."""

    rule_id: str
    title: str
    status: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        """Serialise the rule outcome into a dictionary."""

        return {
            "id": self.rule_id,
            "title": self.title,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class GuardReport:
    """Aggregate output produced by :class:`RiskEngine`."""

    status: str
    rules: list[GuardRuleResult]
    suggested_contracts: int
    account_balance: float
    policy_version: str | None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the report."""

        return {
            "status": self.status,
            "suggested_contracts": self.suggested_contracts,
            "account_balance": self.account_balance,
            "policy_version": self.policy_version,
            "rules": [rule.to_dict() for rule in self.rules],
        }


class RiskEngine:
    """Loads and evaluates the static guard policy."""

    def __init__(self, policy: Mapping[str, Any], policy_path: Path) -> None:
        self._policy = dict(policy)
        self._policy_path = policy_path
        self._rules: list[Mapping[str, Any]] = list(policy.get("rules", []))
        self._defaults: MutableMapping[str, Any] = dict(policy.get("defaults", {}))
        self._profile_defaults: MutableMapping[str, Any] = dict(
            policy.get("profile", {})
        )
        self._policy_metadata = {
            "name": policy.get("name", "Toptek guard policy"),
            "version": policy.get("version"),
            "path": str(policy_path),
        }

    @classmethod
    def from_policy(cls, path: Path | None = None) -> "RiskEngine":
        """Construct the engine from ``policy.yml`` or *path* if provided."""

        if path is None:
            path = Path(__file__).with_name(_POLICY_FILENAME)
        policy_text = path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(policy_text)
        if not isinstance(loaded, Mapping):  # pragma: no cover - defensive guard
            raise ValueError("Guard policy must deserialize to a mapping")
        return cls(loaded, path)

    @property
    def defaults(self) -> Mapping[str, Any]:
        """Return numeric defaults used during evaluation."""

        return self._defaults

    @property
    def policy_metadata(self) -> Mapping[str, Any]:
        """Expose static metadata about the currently loaded policy."""

        return self._policy_metadata

    def _default_float(self, key: str, fallback: float = 0.0) -> float:
        return _coerce_float(self._defaults.get(key), fallback)

    def build_profile(
        self, overrides: Mapping[str, Any] | None = None
    ) -> core_risk.RiskProfile:
        """Create a :class:`~toptek.core.risk.RiskProfile` from defaults + overrides."""

        data: Dict[str, Any] = dict(self._profile_defaults)
        if overrides:
            data.update(overrides)
        restricted_hours = [dict(window) for window in data.get("restricted_hours", [])]
        return core_risk.RiskProfile(
            max_position_size=int(data.get("max_position_size", 1)),
            max_daily_loss=float(data.get("max_daily_loss", 1000)),
            restricted_hours=restricted_hours,
            atr_multiplier_stop=float(data.get("atr_multiplier_stop", 2.0)),
            cooldown_losses=int(data.get("cooldown_losses", 2)),
            cooldown_minutes=int(data.get("cooldown_minutes", 30)),
        )

    def evaluate(
        self,
        profile: core_risk.RiskProfile,
        *,
        account_balance: float | None = None,
        atr: float | None = None,
        tick_value: float | None = None,
        risk_per_trade: float | None = None,
    ) -> GuardReport:
        """Run the configured guard checks for *profile* and return a report."""

        balance_default = self._default_float("account_balance")
        atr_default = self._default_float("atr")
        tick_default = self._default_float("tick_value")
        risk_default = self._default_float("risk_per_trade")
        balance = _coerce_float(account_balance, balance_default)
        atr_value = _coerce_float(atr, atr_default)
        tick = _coerce_float(tick_value, tick_default)
        risk_fraction = _coerce_float(risk_per_trade, risk_default)
        suggested = core_risk.position_size(
            balance,
            profile,
            atr=atr_value,
            tick_value=tick,
            risk_per_trade=risk_fraction,
        )
        results: list[GuardRuleResult] = []
        worst_status = _GUARD_OK
        for rule in self._rules:
            outcome = self._evaluate_rule(rule, profile, suggested)
            results.append(outcome)
            if outcome.status != _GUARD_OK:
                worst_status = _GUARD_DEFENSIVE
        return GuardReport(
            status=worst_status,
            rules=results,
            suggested_contracts=suggested,
            account_balance=balance,
            policy_version=self._policy_metadata.get("version"),
        )

    def render_report(self, report: GuardReport) -> str:
        """Return a human readable representation of *report*."""

        lines = [
            f"Guard policy: {self._policy_metadata.get('name')}"
            f" (version {report.policy_version or 'n/a'})",
            f"Policy file: {self._policy_metadata.get('path')}",
            f"Account balance assumption: ${report.account_balance:,.2f}",
            f"Suggested contracts: {report.suggested_contracts}",
            f"Overall status: {report.status}",
            "",
            "Rule breakdown:",
        ]
        for rule in report.rules:
            lines.append(f"- [{rule.status}] {rule.title}: {rule.message}")
        return "\n".join(lines)

    def _evaluate_rule(
        self,
        rule: Mapping[str, Any],
        profile: core_risk.RiskProfile,
        suggested_contracts: int,
    ) -> GuardRuleResult:
        rule_id = str(rule.get("id", "rule"))
        title = str(rule.get("title", rule_id.replace("-", " ").title()))
        rule_type = str(rule.get("type", "position_size"))
        ok_template = str(rule.get("ok", "Guard ready."))
        defensive_template = str(
            rule.get(
                "defensive",
                "Guard defensive. Stand down and review the playbook before trading.",
            )
        )
        status = _GUARD_OK
        message_kwargs: Dict[str, Any]
        if rule_type == "position_size":
            minimum = int(rule.get("min_contracts", 1))
            status = _GUARD_OK if suggested_contracts >= minimum else _GUARD_DEFENSIVE
            message_kwargs = {"suggested": suggested_contracts, "required": minimum}
        elif rule_type == "max_daily_loss":
            policy_cap = float(rule.get("max_loss", profile.max_daily_loss))
            status = (
                _GUARD_OK if profile.max_daily_loss <= policy_cap else _GUARD_DEFENSIVE
            )
            message_kwargs = {
                "configured": profile.max_daily_loss,
                "policy": policy_cap,
            }
        elif rule_type == "cooldown":
            ceiling = int(rule.get("max_losses", profile.cooldown_losses))
            status = (
                _GUARD_OK if profile.cooldown_losses <= ceiling else _GUARD_DEFENSIVE
            )
            message_kwargs = {
                "configured": profile.cooldown_losses,
                "policy": ceiling,
            }
        else:  # pragma: no cover - reserved for future policy types
            raise ValueError(f"Unsupported guard rule type: {rule_type}")
        template = ok_template if status == _GUARD_OK else defensive_template
        return GuardRuleResult(
            rule_id=rule_id,
            title=title,
            status=status,
            message=_safe_format(template, **message_kwargs),
        )


def _safe_format(template: str, **values: Any) -> str:
    """Format *template* while swallowing placeholder errors."""

    try:
        return template.format(**values)
    except (KeyError, ValueError):  # pragma: no cover - defensive fallback
        return template


def _coerce_float(value: Any, fallback: float) -> float:
    """Best-effort conversion of *value* to ``float`` with a fallback."""

    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the Toptek guard policy and evaluation output.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Optional path to a custom guard policy YAML.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Run the evaluation with policy defaults and print the report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry-point for ``python -m toptek.risk.engine``."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    engine = RiskEngine.from_policy(args.policy)
    profile = engine.build_profile()
    report = engine.evaluate(profile)
    if args.dryrun:
        print(engine.render_report(report))
        return 0
    # Default behaviour mirrors --dryrun to keep the CLI informative.
    print(engine.render_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI module guard
    raise SystemExit(main())
