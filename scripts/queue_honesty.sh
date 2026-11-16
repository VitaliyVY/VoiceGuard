#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1
LOG=results/run12_log.txt; : > $LOG
echo "=== Перевірка на чесність  $(date)" >> $LOG
python scripts/uk_honesty.py >> $LOG 2>&1
echo "exit $?  $(date)" >> $LOG
