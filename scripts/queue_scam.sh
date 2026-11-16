#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
LOG=results/run11_log.txt
echo "=== Переклад (продовження)  $(date)" >> $LOG
python scripts/scam_translate.py --per-type 10 >> $LOG 2>&1
echo "=== Оцінка  $(date)" >> $LOG
python scripts/scam_eval.py >> $LOG 2>&1
echo "exit $?  $(date)" >> $LOG
