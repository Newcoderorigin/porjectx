from toptek.ai_server.tools import (
    BacktestRequest,
    deflated_sharpe_ratio,
    metrics_report,
    run_backtest_tool,
    triple_barrier_labels,
    walk_forward_report,
)


def test_triple_barrier_generates_labels():
    prices = [100 + idx * 0.05 for idx in range(120)]
    labels = triple_barrier_labels(prices, horizon=5, vol_lookback=10)
    assert len(labels) == len(prices) - 1
    assert set(labels).issubset({-1, 0, 1})


def test_backtest_tool_returns_metrics():
    request = BacktestRequest(
        symbol="ES",
        start="2020-01-01",
        end="2020-12-31",
        costs=0.1,
        slippage=0.05,
        vol_target=0.15,
    )
    report = run_backtest_tool(request)
    assert "sharpe" in report
    assert "equity_curve" in report
    assert report["trades"] >= 0


def test_metrics_report_and_dsr():
    pnl = [0.01, -0.02, 0.015, 0.02, -0.005]
    metrics = metrics_report(pnl)
    assert metrics["psr"] >= 0.0
    dsr = deflated_sharpe_ratio(
        metrics["sharpe"], samples=len(pnl), num_trials=3, skewness=0.0, kurtosis=3.0
    )
    assert 0.0 <= dsr <= 1.0


def test_walk_forward_report_shape(tmp_path):
    cfg = tmp_path / "wf.yml"
    cfg.write_text("instrument: ES", encoding="utf-8")
    report = walk_forward_report(str(cfg))
    assert report["config_path"] == str(cfg)
    assert len(report["records"]) == 5
    assert any(item["stress_tests"] for item in [report])
