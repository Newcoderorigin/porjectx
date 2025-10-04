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

## Data schema & storage

The platform persists user sessions, trades, rich events, and model decisions
in `var/toptek.db` (SQLite) using the schema described in
`toptek/data/schema.sql`. Each nightly run can export the same tables to
`data/*.parquet` for modelling parity across research, training, and live
execution. The CLI entry point (`python -m toptek.data.io --init --demo`)
creates the schema and loads 200 demo rows while materialising the Parquet
snapshots.

## Unified feature pipeline

`toptek/features/pipeline.py::build_features` converts any OHLCV frame into
aligned feature matrices, labels, and metadata while caching content hashes to
avoid recomputation. All downstream systems (research notebooks, training
scripts, the backtester, and the UI) call this shared function, ensuring a
single warm-up mask and deterministic feature ordering. Cached artefacts live in
`.cache/` by default but can be configured per run.

## Learning loop & hit-rate optimiser

* `toptek/model/train.py` fits a calibrated scikit-learn classifier, writes
  `model.pkl`, `calibrator.pkl`, and a `model_card.json` containing metrics and
  the feature hash.
* `toptek/model/threshold.py` supplies `opt_threshold`, which maximises hit rate
  under coverage and expected-value constraints.
* `toptek/backtest/run.py` reuses the calibrated model and threshold optimiser
  to filter trades, reporting hit rate, coverage, expectancy, Sharpe ratio, max
  drawdown, and the chosen threshold in JSON form.
* `toptek/loops/learn.py` implements the nightly learning loop: it ingests the
  latest trades/predictions, rebuilds features, retrains the model, and stores a
  dated report in `reports/` detailing metric deltas.

## Runbook

```bash
# 1) Initialise + demo data
python -m toptek.data.io --init --demo

# 2) Train (with calibration) + threshold search
python -m toptek.model.train --config configs/config.yml

# 3) Backtest with calibrated decisions
python -m toptek.backtest.run --config configs/config.yml --use-calibrated --optimize-threshold

# 4) Nightly learning loop (can be scheduled)
python -m toptek.loops.learn --config configs/config.yml

# 5) Tests
pytest -q
```
