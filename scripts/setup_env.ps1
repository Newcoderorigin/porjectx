param(
    [string]$Python = "py -3.11"
)

$ErrorActionPreference = "Stop"

if (Test-Path ".venv") {
    Remove-Item ".venv" -Recurse -Force
}

& $Python -m venv .venv

$venvPath = Join-Path (Resolve-Path ".venv").Path "Scripts"
$venvPython = Join-Path $venvPath "python.exe"

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

$stackCheck = @"
import importlib
import json
import platform

modules = ["numpy", "scipy", "sklearn", "pandas", "joblib", "threadpoolctl"]
versions = {name: importlib.import_module(name).__version__ for name in modules}
print("STACK_OK")
print(json.dumps({
    "python": platform.python_version(),
    "versions": versions,
}, indent=2))
"@

& $venvPython -c $stackCheck
