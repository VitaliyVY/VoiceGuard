#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1
LOG=results/run8_log.txt; : > $LOG
echo "=== Повна система з WavLM  $(date)" >> $LOG
python scripts/tandem.py --cm models/cm_gmm_lfcc_trim_g128.pkl models/cm_gmm_lfcc4k_trim_g128_aug-clean-opus16-telephone-noise20.pkl models/cm_ssl.pkl \
  --feature lfcc lfcc4k ssl --names "A (базовий)" "D (0–4 кГц + аугм.)" "WavLM" --workers 3 --out results/tandem_wavlm.json >> $LOG 2>&1
echo "exit $?  $(date)" >> $LOG
