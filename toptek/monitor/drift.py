"""Data drift diagnostics with PSI and KS severity scoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from collections.abc import Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, List, Optional, Sequence, TypeAlias, cast

try:  # pragma: no cover - optional pandas dependency
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional pandas dependency
    pd = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame as _PandasDataFrame
else:  # pragma: no cover
    _PandasDataFrame = object  # type: ignore[misc, assignment]

ColumnarLike: TypeAlias = Mapping[str, Iterable[object]] | _PandasDataFrame


_EPSILON = 1e-9


class Severity(IntEnum):
    """Severity tiers for drift and latency monitors."""

    STABLE = 0
    WATCH = 1
    ALERT = 2
    UNKNOWN = 3

    @property
    def label(self) -> str:
        labels = {
            Severity.STABLE: "stable",
            Severity.WATCH: "watch",
            Severity.ALERT: "alert",
            Severity.UNKNOWN: "unknown",
        }
        return labels[self]

    def describe(self) -> str:
        descriptions = {
            Severity.STABLE: "Distributions align with reference expectations.",
            Severity.WATCH: "Shifts detected; monitor the feature closely.",
            Severity.ALERT: "Large drift; retraining or investigation required.",
            Severity.UNKNOWN: "Insufficient data to evaluate drift.",
        }
        return descriptions[self]


@dataclass(frozen=True)
class DriftMetric:
    """Container for drift statistics."""

    psi: float
    ks: float


@dataclass(frozen=True)
class DriftFeatureReport:
    """Feature-level drift findings."""

    feature: str
    metric: DriftMetric
    psi_severity: Severity
    ks_severity: Severity
    severity: Severity
    message: str


@dataclass(frozen=True)
class DriftReport:
    """Aggregate drift report for a dataset."""

    features: Mapping[str, DriftFeatureReport]
    overall: Severity
    summary: str


class _Thresholds:
    """Severity thresholds for PSI and KS."""

    PSI = (0.1, 0.25)
    KS = (0.1, 0.2)


def compute_drift_report(
    reference: ColumnarLike,
    current: ColumnarLike,
    *,
    features: Optional[Sequence[str]] = None,
    bins: int = 10,
) -> DriftReport:
    """Compute PSI/KS drift metrics and severity tiers.

    Parameters
    ----------
    reference:
        Historical data establishing the baseline distribution.
    current:
        Recent data to compare against the baseline.
    features:
        Optional subset of columns to analyse. When omitted, the intersection of
        columns present in both datasets is used.
    bins:
        Number of quantile bins used when estimating the PSI.

    Returns
    -------
    DriftReport
        Structured report including per-feature and aggregate severities.
    """

    if bins < 2:
        raise ValueError("`bins` must be >= 2 to compute PSI.")

    reference_columns = _column_names(reference)
    current_columns = _column_names(current)

    if features is None:
        features = sorted(set(reference_columns).intersection(current_columns))
    else:
        missing = [
            col
            for col in features
            if col not in reference_columns or col not in current_columns
        ]
        if missing:
            raise KeyError(
                f"Requested features not available in both datasets: {missing}"
            )

    if not features:
        raise ValueError("No overlapping features to compute drift.")

    per_feature: MutableMapping[str, DriftFeatureReport] = {}
    alerts: List[str] = []
    unknown_features: List[str] = []

    for feature in features:
        ref_values = _extract_numeric(reference, feature)
        cur_values = _extract_numeric(current, feature)

        if not ref_values or not cur_values:
            report = DriftFeatureReport(
                feature=feature,
                metric=DriftMetric(float("nan"), float("nan")),
                psi_severity=Severity.UNKNOWN,
                ks_severity=Severity.UNKNOWN,
                severity=Severity.UNKNOWN,
                message="Insufficient data points to compute drift.",
            )
            per_feature[feature] = report
            unknown_features.append(feature)
            continue

        psi_value = _population_stability_index(ref_values, cur_values, bins=bins)
        ks_value = _kolmogorov_smirnov(ref_values, cur_values)

        psi_severity = _severity_from_value(psi_value, _Thresholds.PSI)
        ks_severity = _severity_from_value(ks_value, _Thresholds.KS)
        severity = max(psi_severity, ks_severity, key=lambda s: s.value)

        if severity == Severity.STABLE:
            message = "No material drift detected."
        elif severity == Severity.WATCH:
            message = "Moderate drift detected; monitor the feature."
        else:
            message = "Severe drift detected; investigate immediately."

        report = DriftFeatureReport(
            feature=feature,
            metric=DriftMetric(float(psi_value), float(ks_value)),
            psi_severity=psi_severity,
            ks_severity=ks_severity,
            severity=severity,
            message=message,
        )

        per_feature[feature] = report

        if severity in (Severity.WATCH, Severity.ALERT):
            alerts.append(f"{feature}: {severity.label}")

    non_unknown = [
        feature_report.severity
        for feature_report in per_feature.values()
        if feature_report.severity != Severity.UNKNOWN
    ]

    overall: Severity
    if non_unknown:
        overall = max(non_unknown, key=lambda s: s.value)
    else:
        overall = Severity.UNKNOWN

    if overall == Severity.STABLE:
        summary = "All monitored features remain stable."
        if unknown_features:
            summary += f" ({', '.join(unknown_features)} missing data)"
    elif overall == Severity.UNKNOWN:
        summary = "One or more features lacked data for drift assessment."
    else:
        summary = "; ".join(alerts)
        if unknown_features:
            summary += f"; {', '.join(unknown_features)} missing data"

    return DriftReport(features=dict(per_feature), overall=overall, summary=summary)


def _column_names(data: ColumnarLike) -> List[str]:
    if pd is not None and isinstance(data, pd.DataFrame):  # type: ignore[arg-type]
        return [str(column) for column in data.columns]
    if isinstance(data, Mapping):
        mapping_data = cast(Mapping[str, Iterable[object]], data)
        return [str(column) for column in mapping_data.keys()]
    raise TypeError("Unsupported container type for drift computation.")


def _extract_numeric(data: ColumnarLike, feature: str) -> List[float]:
    if pd is not None and isinstance(data, pd.DataFrame):  # type: ignore[arg-type]
        series = pd.to_numeric(data[feature], errors="coerce").dropna()
        return [float(value) for value in series.tolist()]
    if isinstance(data, Mapping):
        mapping_data = cast(Mapping[str, Iterable[object]], data)
        if feature not in mapping_data:
            raise KeyError(f"Column '{feature}' not present in dataset.")
        column = mapping_data[feature]
    else:
        raise TypeError("Unsupported container type for drift computation.")

    return _coerce_to_list(column)


def _coerce_to_list(values: Iterable[object]) -> List[float]:
    cleaned: List[float] = []
    for value in values:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        cleaned.append(number)

    return cleaned


def _population_stability_index(
    reference: List[float], current: List[float], *, bins: int
) -> float:
    """Population Stability Index with quantile binning."""

    if not reference or not current:
        return float("nan")

    reference_sorted = sorted(reference)
    current_sorted = sorted(current)

    quantiles = [index / bins for index in range(bins + 1)]
    edges = [_quantile(reference_sorted, q) for q in quantiles]
    edges = _unique_sorted(edges)

    if len(edges) == 1:
        midpoint = edges[0]
        spread = max(
            max((abs(value - midpoint) for value in reference_sorted), default=0.0),
            max((abs(value - midpoint) for value in current_sorted), default=0.0),
        )
        if spread == 0:
            return 0.0
        edges = [midpoint - spread, midpoint + spread]

    low_bound = min(min(reference_sorted), min(current_sorted)) - _EPSILON
    high_bound = max(max(reference_sorted), max(current_sorted)) + _EPSILON
    edges[0] = min(edges[0], low_bound)
    edges[-1] = max(edges[-1], high_bound)

    ref_hist = _histogram(reference_sorted, edges)
    cur_hist = _histogram(current_sorted, edges)

    ref_total = sum(ref_hist)
    cur_total = sum(cur_hist)

    if ref_total == 0 or cur_total == 0:
        return float("nan")

    psi = 0.0
    for ref_count, cur_count in zip(ref_hist, cur_hist):
        ref_ratio = max(ref_count / ref_total, _EPSILON)
        cur_ratio = max(cur_count / cur_total, _EPSILON)
        psi += (cur_ratio - ref_ratio) * math.log(cur_ratio / ref_ratio)

    return psi


def _kolmogorov_smirnov(reference: List[float], current: List[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""

    if not reference or not current:
        return float("nan")

    reference_sorted = sorted(reference)
    current_sorted = sorted(current)

    n_ref = len(reference_sorted)
    n_cur = len(current_sorted)
    ref_index = 0
    cur_index = 0
    max_diff = 0.0

    for value in _unique_sorted(reference_sorted + current_sorted):
        while ref_index < n_ref and reference_sorted[ref_index] <= value:
            ref_index += 1
        while cur_index < n_cur and current_sorted[cur_index] <= value:
            cur_index += 1
        ref_cdf = ref_index / n_ref
        cur_cdf = cur_index / n_cur
        diff = abs(ref_cdf - cur_cdf)
        if diff > max_diff:
            max_diff = diff

    return max_diff


def _quantile(sorted_values: List[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty data.")
    if quantile <= 0:
        return sorted_values[0]
    if quantile >= 1:
        return sorted_values[-1]

    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[int(position)]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _histogram(values: List[float], edges: List[float]) -> List[int]:
    counts = [0 for _ in range(len(edges) - 1)]
    for value in values:
        for index in range(len(counts)):
            left = edges[index]
            right = edges[index + 1]
            if index == len(counts) - 1:
                if left <= value <= right:
                    counts[index] += 1
                    break
            elif left <= value < right:
                counts[index] += 1
                break
    return counts


def _unique_sorted(values: Iterable[float]) -> List[float]:
    unique: List[float] = []
    for value in sorted(values):
        if not unique or not math.isclose(
            value, unique[-1], rel_tol=0.0, abs_tol=_EPSILON
        ):
            unique.append(value)
    return unique


def _severity_from_value(value: float, thresholds: Sequence[float]) -> Severity:
    """Map a metric value to a severity tier."""

    if value is None or math.isnan(value):
        return Severity.UNKNOWN

    low, high = thresholds
    if value < low:
        return Severity.STABLE
    if value < high:
        return Severity.WATCH
    return Severity.ALERT
