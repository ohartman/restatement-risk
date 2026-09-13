#!/bin/bash
PY=${PY:-python}
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "################ financial data sets 2024-2025 ($(date +%H:%M)) ################"
$PY -u fetch_sec.py 2024 2025 2>&1 | grep --line-buffered -v Warning
echo "################ form indices and insider data sets ($(date +%H:%M)) ################"
$PY -u live/fetch_inputs.py 2>&1 | grep --line-buffered -v Warning
echo "################ daily indices 2025 ($(date +%H:%M)) ################"
$PY -u fetch_daily_index.py 2025 2025 8 2>&1 | grep --line-buffered -v "Warning\|403"
echo "################ 8-K events 2024-2025 ($(date +%H:%M)) ################"
$PY -u fetch_events.py 4.01 2024 2025 2>&1 | grep --line-buffered -v Warning
$PY -u fetch_events.py 5.02 2024 2025 2>&1 | grep --line-buffered -v Warning
echo "LIVE FETCH DONE ($(date +%H:%M))"
