#!/bin/bash
# Повторні навчання LCNN з іншими початковими значеннями (розкид результату)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1
CH="clean opus32 opus16 opus8 telephone amrnb amrwb gsm noise20 noise10"
LOG=results/run6_log.txt; : > $LOG
for SEED in 1 2; do
  echo "=== CNN lfcc4k, аугментація, seed $SEED  $(date)" >> $LOG
  python scripts/cm_cnn.py --feature lfcc4k --channels $CH --train-channels clean opus16 telephone noise20 --seed $SEED >> $LOG 2>&1
done
echo "exit $?  $(date)" >> $LOG
