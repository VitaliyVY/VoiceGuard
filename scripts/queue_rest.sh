#!/bin/bash
# Черга експериментів, що лишилися (запускається як незалежний процес Windows)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1
CH="clean opus32 opus16 opus8 telephone amrnb amrwb gsm noise20 noise10"
LOG=results/run5_log.txt; : > $LOG
echo "=== GMM lfcc4k, аугментація  $(date)" >> $LOG
python scripts/cm_gmm.py --feature lfcc4k --trim --components 128 --dev-per-attack 200 --eval-per-attack 200 \
  --workers 6 --channels $CH --train-channels clean opus16 telephone noise20 >> $LOG 2>&1
echo "=== CNN lfcc4k, аугментація  $(date)" >> $LOG
python scripts/cm_cnn.py --feature lfcc4k --channels $CH --train-channels clean opus16 telephone noise20 >> $LOG 2>&1
echo "=== CNN lfcc4k, чисті  $(date)" >> $LOG
python scripts/cm_cnn.py --feature lfcc4k --channels $CH >> $LOG 2>&1
echo "exit $?  $(date)" >> $LOG
