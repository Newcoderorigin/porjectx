# Toptek Starter

## Overview

Toptek is a Windows-friendly starter kit for working with the ProjectX Gateway (TopstepX) to research futures markets, engineer features, train simple models, backtest ideas, and manage paper/live trading from a single interface. It combines a Tkinter GUI with a CLI for automation-friendly workflows.

> **Not financial advice. Manual trading decisions only. Always respect Topstep rules and firm risk limits.**

## Quickstart

```powershell
# Windows, Python 3.10-3.13 (example uses 3.12)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-lite.txt

copy .env.example .env
# edit PX_* in .env OR use GUI Settings
python main.py
```

### Web console build steps

The React-based markets console lives under `toptek/ui/web`. Build the production
assets once before launching the Tkinter shell with the web tab enabled:

```powershell
cd toptek/ui/web
npm install
npm run build
```

Enable the packaged assets by setting `web_frontend.enabled: true` in
`config/app.yml` or via the `.env` override. When the assets are present the GUI
exposes an **Open Web Console** button that serves the bundle from an embedded
HTTP server.

### TradingView integration

Toptek ships with a lightweight TradingView launcher that keeps browser charts
in sync with the GUI symbol/timeframe selections. Enable it in
`config/app.yml` under the `tv` block or via the matching `.env` keys:

```yaml
tv:
  enabled: true              # or TOPTEK_TV_ENABLED=1
  symbol: ES=F               # default chart symbol (TOPTEK_TV_SYMBOL)
  interval: 5m               # default interval (TOPTEK_TV_INTERVAL)
  theme: dark                # dark/light theme (TOPTEK_TV_THEME)
  locale: en                 # locale code (TOPTEK_TV_LOCALE)
  tabs:                      # per-tab toggles (TOPTEK_TV_TAB_*)
    notebook: true           # dedicated TradingView tab
    research: true           # Research tab shortcut
    trade: true              # Trade tab shortcut
  favorites:
    - label: "ES · 5m"
      symbol: ES=F
      interval: 5m
```

The dedicated **TradingView** tab (plus the shortcuts in the Research and Trade
tabs) launch a browser window using these defaults. `Ctrl+Shift+T` works across
all tabs and reuses the most recent parameters. Populate the optional
`tv.favorites` list to pre-load quick presets — each entry accepts `symbol`,
`interval`, and an optional `label` for the dropdown menu.

Research tab launches inherit the current symbol/timeframe fields so your chart
follows the data you just pulled. The Trade tab uses the persisted
TradingView defaults (or the last launch) so guard checks and browser charts
stay aligned. Every launch is logged to the session state for quick reference.

> **TradingView ToS reminder:** their widgets and charts require attribution and
> human verification before placing trades. Keep the attribution footer visible
> and confirm every symbol/interval manually. The launcher simply opens the
> TradingView chart in your browser — no automated trading is performed.

## CLI usage examples

```powershell
python main.py --cli train --symbol ESZ5 --timeframe 5m --lookback 90d
python main.py --cli backtest --symbol ESZ5 --timeframe 5m --start 2025-01-01
python main.py --cli paper --symbol ESZ5 --timeframe 5m
python -m toptek.replay.sim data/sessions/es_sample.parquet --speed 2.0
```

## Project structure

```
toptek/
  main.py
  README.md
  requirements-lite.txt
  requirements-streaming.txt
  .env.example
  config/
    app.yml
    risk.yml
    features.yml
  replay/
    sim.py
  core/
    gateway.py
    symbols.py
    data.py
    features.py
    model.py
    backtest.py
    risk.py
    live.py
    utils.py
  monitor/
    drift.py
    latency.py
  gui/
    app.py
    widgets.py
```

## Configuration

Configuration defaults live under the `config/` folder and are merged with values from `.env`. Use the GUI Settings tab (Login section) to create or update the `.env` file if one is missing.

## Requirements profiles

- `requirements-lite.txt`: minimal dependencies for polling workflows. Pins the numeric stack to NumPy `>=2.1.2,<2.3`, SciPy `>=1.14.1,<1.16`, and scikit-learn `==1.6.0` so the wheels vetted in `STACK_REQUIREMENTS` (and reiterated in the root README) remain importable across Python 3.10-3.13.
- `requirements-streaming.txt`: extends the lite profile with optional SignalR streaming support.
- On start-up `python -m toptek.main` validates that NumPy/SciPy/scikit-learn match the vetted wheels and raises a friendly
  guidance error if the environment drifts. Reinstall with `pip install -r requirements-lite.txt` to resolve mismatches.

## Monitoring helpers

Use the `toptek.monitor` package to keep an eye on data quality and feed
freshness:

- `compute_drift_report` compares PSI/KS statistics between reference and
  current windows, returning severity tiers per feature and overall so the GUI
  can escalate drift badges deterministically.
- `build_latency_badge` maps the latest bar timestamp to friendly status copy
  (`Live`, `Lagging`, `Stalled`) based on configurable thresholds, making it
  trivial to render latency pills in the dashboard header.

## Development notes

  - Source code is fully typed and documented with docstrings.
  - HTTP interactions with ProjectX Gateway rely on `httpx` with retry-once semantics for authentication failures.
  - Feature engineering uses `numpy` and `ta` indicators; additional features can be added to `core/features.py`.
  - Models are persisted locally in the `models/` folder.
  - The replay simulator (`toptek/replay/sim.py`) streams recorded bars into the GUI Replay tab and a standalone CLI (`python -m toptek.replay.sim`).

## Safety

- Symbol validation ensures only CME/CBOT/NYMEX/COMEX futures are traded.
- Risk limits derive from `config/risk.yml` and the GUI enforces Topstep-style guardrails.
- No trading activity occurs automatically; all orders require manual confirmation.

## Optional streaming

Install the streaming extras when ready to experiment with SignalR real-time data:

```powershell
pip install -r requirements-streaming.txt
```

Streaming helpers are stubbed in `core/live.py` and disabled unless `signalrcore` is installed.

## Replay simulator

Use the Replay tab inside the GUI to connect a CSV/Parquet dataset to the live chart. Playback controls expose start, pause,
resume, and seek, while a speed selector lets you accelerate or slow the stream. The same engine powers a CLI entry point:

```bash
python -m toptek.replay.sim path/to/session.parquet --speed 4 --format parquet
```

The CLI prints each bar as JSON and supports seeking via index or timestamp (`--seek 2024-01-15T14:30:00`). The GUI stores the
most recent playback state in the in-memory config so mission control can resume from the same dataset.

