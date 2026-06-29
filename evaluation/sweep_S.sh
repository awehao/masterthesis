#!/usr/bin/env bash
# Sweep the input-smoothness weight S over {0,8,15,22,30}, 5 gmpc_scan trials
# each, to find the S that minimises arrival time / path length while keeping
# success + smoothness. For each S: edit gmpc_params -> rebuild -> run 5 ->
# save results to evaluation/results/sweep/S_<val>.csv. Restores S=15/15/8 at end.
#
# Usage:  ./evaluation/sweep_S.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

PARAMS=src/ammr_wholebody_mpc/config/gmpc_params.yaml
OUT=evaluation/results/sweep
mkdir -p "$OUT"

# back up whatever gmpc_scan.csv exists now (e.g. the benchmark run) — the sweep
# overwrites it per-S.
[ -f evaluation/results/omnibot_dynamic_gmpc_scan.csv ] && \
  cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "$OUT/_pre_sweep_gmpc_scan.csv"

set_S () {   # $1 = S value; S_w kept at 0.53*S (matches the 15->8 standard ratio)
  local s="$1"
  local sw
  sw=$(awk "BEGIN{printf \"%.1f\", $s*0.53}")
  sed -i "s/^    S_vx:.*/    S_vx:               ${s}.0/;
          s/^    S_vy:.*/    S_vy:               ${s}.0/;
          s/^    S_w:.*/    S_w:                ${sw}/" "$PARAMS"
}

for S in 0 8 15 22 30; do
  echo "========================= S = $S ========================="
  set_S "$S"
  colcon build --packages-select ammr_wholebody_mpc 2>&1 | tail -1
  source install/setup.bash
  ./evaluation/run_omnibot_dynamic.sh 5 250 17 17 gmpc_scan
  cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "$OUT/S_${S}.csv"
  echo "saved -> $OUT/S_${S}.csv"
done

# restore standard config S=15/15/8
sed -i "s/^    S_vx:.*/    S_vx:               15.0/;
        s/^    S_vy:.*/    S_vy:               15.0/;
        s/^    S_w:.*/    S_w:                 8.0/" "$PARAMS"
colcon build --packages-select ammr_wholebody_mpc 2>&1 | tail -1
echo "DONE. swept S in {0,8,15,22,30} -> $OUT/S_*.csv ; restored S=15/15/8"
