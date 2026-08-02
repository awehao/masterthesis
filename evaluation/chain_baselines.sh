#!/usr/bin/env bash
# MPPI and RPP baselines on bigarena, after the static-gate chain finishes.
#
# omni_bot_baseline.launch.py was hardcoded to random_room and to a (0,0) spawn,
# so the baselines could not run the scenario the GMPC numbers come from -- any
# comparison table would have mixed two different worlds. It now takes the same
# BIGARENA / ARENA / TRAJ selection and the same SPAWN_X/SPAWN_Y as the dynamic
# launch, and carries the three NVIDIA EGL variables (without them gz renders
# through Mesa and the gpu_lidar returns nothing usable).
#
# The baselines keep their own costmap tuning (inflation 0.45) and their raw
# /scan input: they have no CBF, so the costmap is the only thing that can see a
# mover. Giving them /scan_filtered would blind them on purpose. Each method
# therefore runs at its own best, which is the fair comparison.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid

echo "[$(date +%T)] waiting for the static-gate chain ..."
while ! grep -q "chain complete" evaluation/logs/chainStatic.log 2>/dev/null; do sleep 120; done
while [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; do sleep 60; done
sleep 30
echo "[$(date +%T)] static chain done"

# Build only now: colcon swaps install/ under any running node.
source /opt/ros/jazzy/setup.bash
colcon build --packages-select my_omnibot_description \
    > evaluation/logs/build_baseline.log 2>&1
grep -q "SPAWN_X" install/my_omnibot_description/share/my_omnibot_description/launch/omni_bot_baseline.launch.py \
    || { echo "FATAL: baseline launch not installed"; exit 1; }
echo "[$(date +%T)] build ok"

run_baseline () {               # $1 = mppi | rpp
    local m="$1"
    local csv="evaluation/results/omnibot_dynamic_${m}.csv"
    echo "[$(date +%T)] === smoke test $m ==="
    POSES_CSV="$PWD/evaluation/results/bigarena_poses.csv" \
    BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        timeout 500 ./evaluation/run_omnibot_dynamic.sh 1 150 0 0 "$m" \
        > "evaluation/logs/smoke_${m}.log" 2>&1
    local moved
    moved=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="path_length_m") c=i; next}
                     NR==2{print ($c+0>0.5)?"1":"0"; exit}' "$csv" 2>/dev/null)
    if [ "$moved" != "1" ]; then
        echo "[$(date +%T)] $m SMOKE FAILED (robot did not move) -- skipping this baseline"
        return 1
    fi
    echo "[$(date +%T)] $m smoke ok"

    echo "[$(date +%T)] === baseline $m (30 trials) ==="
    mkdir -p "evaluation/results/base_${m}"
    POSES_CSV="$PWD/evaluation/results/bigarena_poses.csv" \
    BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 "$m" \
        > "evaluation/logs/base_${m}.log" 2>&1
    cp "$csv" "evaluation/results/base_${m}/batch.csv"

    local ad="evaluation/bags/archive_base_${m}"; mkdir -p "$ad"
    for d in evaluation/bags/${m}__${m}_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    echo "[$(date +%T)] $m done -> evaluation/results/base_${m}/batch.csv (bags $(ls "$ad" | wc -l))"
}

run_baseline mppi
run_baseline rpp
echo "[$(date +%T)] baselines complete"
