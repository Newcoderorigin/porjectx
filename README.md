# porjectx

## Dependency compatibility

The `toptek/requirements-lite.txt` file pins the scientific stack to keep it
compatible with the bundled scikit-learn release:

- `scikit-learn==1.6.0`
- `numpy>=2.1.2,<2.3`
- `scipy>=1.14.1,<1.16`

These ranges follow the support window published by scikit-learn 1.6.x and are
also consumed transitively by `toptek/requirements-streaming.txt` through its
`-r requirements-lite.txt` include. Installing within these bounds avoids the
ABI mismatches that occur with the NumPy/SciPy wheels when using newer major
releases. In particular, upgrading NumPy beyond `<2.3` or SciPy beyond `<1.16`
risks diverging from the tested wheel set, mirroring the guidance already
documented in `toptek/README.md`.

## Verifying the environment

Use Python **3.10 through 3.13**—matching the guidance in `toptek/README.md`'s
quickstart—to stay within the wheel support window for SciPy and
scikit-learn 1.6.x. The refreshed NumPy/SciPy releases ship wheels for CPython
3.13, unlocking the latest interpreter without requiring source builds. Create
and activate a compatible virtual environment, then install and check for
dependency issues:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r toptek/requirements-lite.txt
pip check
```

The final `pip check` call should report "No broken requirements found",
confirming that the pinned dependency set resolves without conflicts. Users on
interpreters older than Python 3.10 or newer than 3.13 should match one of the
supported versions or wait for a future dependency refresh before proceeding.

## UI configuration surface

The manual trading shell and Tkinter dashboard now read defaults from
`configs/ui.yml`. The file ships with sensible demo values so the GUI renders
without external data sources:

- `appearance` &mdash; theme token (currently `dark`) and accent family used by the
  style registry.
- `shell` &mdash; defaults for the research symbol/timeframe, training lookback,
  calibration flag, simulated backtest window, and preferred playbook.
- `chart` &mdash; LiveChart refresh cadence (`fps`), point budget, and price
  precision used by upcoming streaming widgets.
- `status` &mdash; copy for the status banners shown in the Login, Train, Backtest,
  and Guard tabs so product teams can retune messaging without touching code.

Operators can override the YAML at runtime either with environment variables or
CLI flags:

- Environment variables follow the `TOPTEK_UI_*` convention, e.g.
  `TOPTEK_UI_SYMBOL`, `TOPTEK_UI_INTERVAL`, `TOPTEK_UI_LOOKBACK_BARS`,
  `TOPTEK_UI_CALIBRATE`, `TOPTEK_UI_FPS`, and `TOPTEK_UI_THEME`.
- CLI switches (`--symbol`, `--timeframe`, `--lookback`, `--model`, `--fps`)
  apply the same overrides for one-off runs and are reflected back into the GUI
  when it launches.

These controls keep the default Topstep demo intact while making it easy to
point the toolkit at alternative markets or stress-test higher frequency charts
without editing source files.

## Risk guard engine

The `toptek/risk` package now ships a policy-driven guard engine that powers
the trade tab badge and the `Ctrl+P` panic circuit. Run

```bash
python -m toptek.risk.engine --dryrun
```

to inspect the aggregated guard report and rule breakdown without launching the
GUI. Install [`PyYAML`](https://pyyaml.org/) if the command reports a missing
dependency. The same report is serialised back into the `configs["trade"]`
dictionary whenever the guard is refreshed, allowing downstream automation to
respond when the status shifts between `OK` and `DEFENSIVE_MODE`.

## Monitoring surface

Operational dashboards can now import helpers from `toptek.monitor` to surface
data quality and feed health at a glance:

- `toptek.monitor.compute_drift_report` evaluates PSI/KS drift across a
  DataFrame slice, returning feature-level and aggregate severities that the UI
  can render as badges or alerts.
- `toptek.monitor.build_latency_badge` converts a timestamp for the latest bar
  into deterministic status copy (`Live`, `Lagging`, `Stalled`) based on latency
  thresholds.

Both utilities return frozen dataclasses to keep the API predictable for
widgets, scripts, or automated monitors.

## AI Server & Quant Co-Pilot

The `toptek.ai_server` package launches a local FastAPI application that
coordinates LM Studio, auto-selects the best local model, and exposes streaming
chat and quant tooling endpoints:

- `GET /healthz` &mdash; verifies LM Studio is reachable and that quant utilities
  load successfully.
- `GET /models` &mdash; lists discovered LM Studio models alongside throughput
  telemetry (TTFT/tokens-per-second) gathered during use.
- `POST /models/select` &mdash; overrides the auto-router and pins a default model
  for subsequent chats.
- `POST /chat` &mdash; streams an OpenAI-compatible conversation via the chosen
  model while updating router telemetry.
- `POST /tools/backtest`, `/tools/walkforward`, `/tools/metrics` &mdash; bridges to
  local quant routines for deterministic analysis and risk controls.

Launch the service with the helper script, optionally pointing at a custom
configuration or overriding environment variables (see `configs/ai.yml` and
`toptek/.env.example`):

```bash
scripts/start_ai_server.sh --port 8080
```

The server will attempt to auto-start LM Studio with `lms server start` if it is
not already running on `LMSTUDIO_BASE_URL`. Set `LMSTUDIO_AUTO_START=false` to
skip this behaviour. The bundled Tailwind/htmx UI is available at `/` and
includes live chat, a model picker, and buttons that exercise the quant tools.
