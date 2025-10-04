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
