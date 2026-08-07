#!/usr/bin/env bash
# End-to-end: generate data -> build warehouse -> validity gate -> analyze
set -euo pipefail

python -m data_gen.simulate --streams 2000 --seed 42
DBT_PROFILES_DIR=. dbt build
python -m analysis.validity        # gate: must pass before trusting results
python -m analysis.power
python -m analysis.effect
echo
echo "Dashboard:  streamlit run dashboard/app.py"