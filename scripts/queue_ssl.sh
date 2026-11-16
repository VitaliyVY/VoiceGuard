#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1
LOG=results/run7_log.txt; : > $LOG
echo "=== WavLM: ознаки + класифікатор  $(date)" >> $LOG
python scripts/ssl_cm.py --stage all >> $LOG 2>&1
echo "exit $?  $(date)" >> $LOG
