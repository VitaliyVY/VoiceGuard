#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1
rm -f results/tandem_cm_scores_ssl.csv
LOG=results/run9_log.txt; : > $LOG
python scripts/tandem.py --cm models/cm_ssl.pkl --feature ssl --names "WavLM" --workers 3 --out results/tandem_wavlm_scores.json >> $LOG 2>&1
echo "exit $?  $(date)" >> $LOG
