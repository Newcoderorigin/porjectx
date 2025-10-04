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
releases. In particular, upgrading NumPy beyond `<1.28` triggers SciPy's
runtime guard:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x…
downgrade to numpy<2 or rebuild the module.
```

This surfaces as an `ImportError` when SciPy loads compiled extensions,
matching the warning already called out in `toptek/README.md`.

## Verifying the environment

Create and activate a Python 3.10+ virtual environment, then install and check
for compatibility issues:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r toptek/requirements-lite.txt
pip check
python - <<'PY'
import numpy, scipy
print(f"NumPy {numpy.__version__}, SciPy {scipy.__version__} ready")
PY
```

The final `pip check` call should report "No broken requirements found", and
the import smoke test should print the installed versions without raising
errors—confirming that the pinned dependency set resolves without conflicts and
that SciPy can load its compiled modules.
