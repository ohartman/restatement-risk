#!/bin/bash
# Build the full feature stack for 10-Ks filed 2024-2025 in the live/ working directory.
# Every builder resolves data/raw and data/out relative to the working directory, so live/
# holds only the new quarterly data sets (hard links) plus junctions to the shared inputs.
PY=${PY:-python}
MAIN=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$MAIN PYTHONIOENCODING=utf-8 EC_ROOT=$MAIN/live
# run live/fetch_all.sh first (it fetches the 2024-2025 raw inputs)
cd $MAIN/live
for z in $MAIN/data/raw/2024q?.zip $MAIN/data/raw/2025q?.zip; do
  [ -f "$z" ] && [ ! -f "data/raw/$(basename $z)" ] && cmd //c "mklink /H data\\raw\\$(basename $z) $(cygpath -w $z)" > /dev/null
done
cp $MAIN/data/raw/events_401.jsonl $MAIN/data/raw/events_502.jsonl data/raw/
ls data/raw/*.zip
echo "################ 28 ratios / labels panel ($(date +%H:%M)) ################"
$PY -c "from tune import load; X, y, filed, f = load(1095); print(X.shape, filed.min(), filed.max())" 2>&1 | grep -v Warning
echo "################ raw items, letters, events, insider ($(date +%H:%M)) ################"
$PY -u $MAIN/raw_items.py 2>&1 | grep -v Warning | tail -2
$PY -u $MAIN/raw_items2.py 2>&1 | grep -v Warning | tail -2
$PY -u $MAIN/letters_features.py 2>&1 | grep -v "Warning\|^  form_" | tail -3
$PY -u $MAIN/event_features.py 2>&1 | grep -v Warning | tail -2
$PY -u $MAIN/insider_features.py 2>&1 | grep -v "Warning\|_form345" | tail -2
echo "################ history columns ($(date +%H:%M)) ################"
$PY -u $MAIN/scrutiny_features.py 2>&1 | grep -v Warning | tail -3
echo "################ fetch the 10-K documents ($(date +%H:%M)) ################"
$PY -u $MAIN/fetch_all_10k.py 2>&1 | grep -v Warning | tail -3
echo "################ text flags, our cutter ($(date +%H:%M)) ################"
$PY -u $MAIN/scrutiny_v3_clean.py 2>&1 | grep -v Warning | tail -2
echo "################ EDGAR-CRAWLER sections and the thirty flags ($(date +%H:%M)) ################"
$PY -u $MAIN/ec_extract.py --procs 14 2>&1 | grep -v "Warning\|doc_report" | tail -2
$PY -u $MAIN/ec_features.py 2>&1 | grep -v Warning | tail -4
echo "LIVE BUILD DONE ($(date +%H:%M))"
