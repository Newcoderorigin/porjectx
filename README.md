# porjectx

## Reproducible environment bootstrap

The numeric stack is pinned via [`constraints.txt`](constraints.txt) to avoid the
ABI/runtime mismatches that previously caused NumPy/SciPy import errors. The
root [`requirements.txt`](requirements.txt) includes that constraint file and
pulls in the toolkit's dependencies from `toptek/requirements-lite.txt`.

On Windows, run the helper script to recreate a clean environment and verify the
stack:

```powershell
.\scripts\setup_env.ps1
```

The script rebuilds `.venv`, installs from `requirements.txt`, then prints
`STACK_OK` followed by the resolved versions in JSON form. The check ensures the
runtime matches `numpy==1.26.4`, `scipy==1.10.1`, and `scikit-learn==1.3.2`
exactly alongside compatible `pandas`, `joblib`, and `threadpoolctl` wheels.

For POSIX shells the equivalent manual steps are:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Runtime telemetry and guardrails

The entry point now executes `toptek.core.utils.assert_numeric_stack()` and
`toptek.core.utils.set_seeds(42)` during startup. Version validation writes a
structured report to `reports/run_stack.json` so crash reports include the exact
Python and numeric-library versions. Structured logging is initialised via
`logging.basicConfig` with a rotating file handler targeting
`logs/toptek_YYYYMMDD.log` alongside console output, keeping telemetry for both
CLI and GUI sessions.

## UI configuration surface

The manual trading shell and Tkinter dashboard read defaults from
[`configs/ui.yml`](configs/ui.yml). The file ships with sensible demo values so
the GUI renders without external data sources:

- `appearance` — theme token (currently `dark`) and accent family used by the
  style registry.
- `shell` — defaults for the research symbol/timeframe, training lookback,
  calibration flag, simulated backtest window, and preferred playbook.
- `chart` — LiveChart refresh cadence (`fps`), point budget, and price
  precision used by streaming widgets.
- `status` — copy for the status banners shown in the Login, Train, Backtest,
  and Guard tabs so product teams can retune messaging without touching code.

Operators can override the YAML at runtime with environment variables or CLI
flags:

- Environment variables follow the `TOPTEK_UI_*` convention, e.g.
  `TOPTEK_UI_SYMBOL`, `TOPTEK_UI_INTERVAL`, `TOPTEK_UI_LOOKBACK_BARS`,
  `TOPTEK_UI_CALIBRATE`, `TOPTEK_UI_FPS`, and `TOPTEK_UI_THEME`.
- CLI switches (`--symbol`, `--timeframe`, `--lookback`, `--model`, `--fps`)
  apply the same overrides for one-off runs and are reflected back into the GUI
  when it launches.

These controls keep the default Topstep demo intact while making it easy to
point the toolkit at alternative markets or stress-test higher frequency charts
without editing source files.
