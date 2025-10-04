# Toptek Starter

## Overview

Toptek is a Windows-friendly starter kit for working with the ProjectX Gateway (TopstepX) to research futures markets, engineer features, train simple models, backtest ideas, and manage paper/live trading from a single interface. It combines a Tkinter GUI with a CLI for automation-friendly workflows.

> **Not financial advice. Manual trading decisions only. Always respect Topstep rules and firm risk limits.**

## Quickstart

```powershell
# Windows, Python 3.11
.\scripts\setup_env.ps1
.venv\Scripts\activate

copy toptek\.env.example .env
# edit PX_* in .env OR use GUI Settings
python toptek\main.py
```

## CLI usage examples

```powershell
python main.py --cli train --symbol ESZ5 --timeframe 5m --lookback 90d
python main.py --cli backtest --symbol ESZ5 --timeframe 5m --start 2025-01-01
python main.py --cli paper --symbol ESZ5 --timeframe 5m
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
  gui/
    app.py
    widgets.py
```

## Configuration

Configuration defaults live under the `config/` folder and are merged with values from `.env`. Use the GUI Settings tab (Login section) to create or update the `.env` file if one is missing.

## Requirements profiles

- `../constraints.txt`: pins NumPy 1.26.4, SciPy 1.10.1, scikit-learn 1.3.2 plus compatible `pandas`, `joblib`, and `threadpoolctl` wheels.
- `../requirements.txt`: references the constraint file and pulls in the lite dependency set.
- `requirements-lite.txt`: minimal dependencies for polling workflows (consumed via the root requirements).
- `requirements-streaming.txt`: extends the lite profile with optional SignalR streaming support.

## Development notes

- Source code is fully typed and documented with docstrings.
- HTTP interactions with ProjectX Gateway rely on `httpx` with retry-once semantics for authentication failures.
- Feature engineering uses `numpy` and `ta` indicators; additional features can be added to `core/features.py`.
- Models are persisted locally in the `models/` folder.

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

