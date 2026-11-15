#!/usr/bin/env bash
# Chạy 5 task HO-Cap x các mode (cần server Pi-0: bash $HOT3D_WS/scripts/serve_pi0.sh &)
#   bash scripts/07_batch.sh [task ...]
cd "$(dirname "$0")/.."
source env_lab.sh
TASKS=${@:-hocap_0000_G18_3 hocap_0021_G20_4 hocap_0005_G19_3 hocap_0002_G10_2 hocap_0023_G21_2}
mkdir -p outputs/demodiffusion/logs
for t in $TASKS; do
  for m in "retarget" "demodiffusion --time_denoise 0.2" "pi0"; do
    tag=$(echo $m | tr -d ' -' | sed 's/time_denoise/_t/')
    timeout 1800 python scripts/06_demodiffusion_sim.py --task $t --mode $m > outputs/demodiffusion/logs/${t}_${tag}.txt 2>&1
    echo "$t [$m] $(grep -oE 'success=.*|Traceback.*' outputs/demodiffusion/logs/${t}_${tag}.txt | head -1)"
  done
done
