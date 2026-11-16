#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1
until [ -f data/uk/fake/index.csv ]; do sleep 20; done
LOG=results/run10_log.txt; : > $LOG
echo "=== Оцінювання на українських записах  $(date)" >> $LOG
python scripts/uk_eval.py >> $LOG 2>&1
echo "exit $?  $(date)" >> $LOG
