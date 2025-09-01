@"
# Tyre PM Modelling

Runs a LightGBM baseline and a gated TCN with SSL warmup.
Outputs go to `ml_charts/` and `models/`.

## Quick start (Windows/PowerShell)

```powershell
# 1) create venv (first time only)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) install deps
pip install -r requirements.txt

# 3) put your CSV here:
#    data/raw/1.csv
# 4) run
python .\scripts\run_all.py
