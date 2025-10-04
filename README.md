# porjectx

## Dependency compatibility

The `toptek/requirements-lite.txt` file pins the scientific stack to keep it
compatible with the bundled scikit-learn release:

- `scikit-learn==1.3.2`
- `numpy>=1.21.6,<1.28`
- `scipy>=1.7.3,<1.12`

These ranges follow the support window published by scikit-learn 1.3.x and are
also consumed transitively by `toptek/requirements-streaming.txt` through its
`-r requirements-lite.txt` include. Installing within these bounds avoids the
ABI mismatches that occur with the NumPy/SciPy wheels when using newer major
releases. In particular, upgrading NumPy beyond `<1.28` causes SciPy to raise
its "compiled against NumPy 1.x" `ImportError`, mirroring the guidance already
documented in `toptek/README.md`.

## Verifying the environment

Use Python **3.10 or 3.11**—matching the guidance in `toptek/README.md`'s
quickstart—to stay within the wheel support window for SciPy and
scikit-learn 1.3.x. Python 3.12 is currently unsupported because prebuilt
SciPy/scikit-learn wheels for that interpreter depend on NumPy ≥1.28 and
SciPy ≥1.12, which exceed this project's pinned ranges. Create and activate a
compatible virtual environment, then install and check for dependency issues:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r toptek/requirements-lite.txt
pip check
```

The final `pip check` call should report "No broken requirements found",
confirming that the pinned dependency set resolves without conflicts. Users on
Python 3.12 should downgrade to Python 3.10/3.11 or wait for a dependency
refresh that supports NumPy ≥1.28 and SciPy ≥1.12 before proceeding.

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
