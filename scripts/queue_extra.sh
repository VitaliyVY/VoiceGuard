#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1
LOG=results/run13_log.txt; : > $LOG
echo "=== Синтез adapt_mms  $(date)" >> $LOG
python scripts/uk_fake_mms_adapt.py --n 118 2>&1 | grep -v -i "warn\|Loading weights" >> $LOG
echo "=== Додаткові експерименти  $(date)" >> $LOG
python scripts/uk_extra.py >> $LOG 2>&1
echo "exit $?  $(date)" >> $LOG
