#!/usr/bin/env bash
# RPP baseline, retried. Runs after the horizon sweep.
#
# The first attempt was skipped because its single smoke trial failed bring-up
# with "/amcl_pose alive: TIMEOUT after 87 s". That was a transient: MPPI came
# up on the same launch, same world and same map minutes earlier, and the amcl
# blocks of nav2_baseline_mppi.yaml and nav2_baseline_rpp.yaml are identical --
# diffed, no differences at all, not even in the section list. One failed
# attempt was not enough evidence to skip a whole baseline, so the smoke test
# now retries.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
CSV=evaluation/results/omnibot_dynamic_rpp.csv

echo "[$(date +%T)] waiting for the horizon sweep ..."
while ! grep -qE "horizon sweep complete|ABORT" evaluation/logs/chainHoriz.log 2>/dev/null; do sleep 120; done
while [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; do sleep 60; done
sleep 30

ok=0
for try in 1 2 3; do
    echo "[$(date +%T)] === rpp smoke attempt $try ==="
    POSES_CSV="$PWD/evaluation/results/bigarena_poses.csv" \
    BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        timeout 500 ./evaluation/run_omnibot_dynamic.sh 1 150 0 0 rpp \
        > "evaluation/logs/smoke_rpp_${try}.log" 2>&1
    moved=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="path_length_m") c=i; next}
                     NR==2{print ($c+0>0.5)?"1":"0"; exit}' "$CSV" 2>/dev/null)
    if [ "$moved" = "1" ]; then ok=1; echo "[$(date +%T)] rpp smoke ok on attempt $try"; break; fi
    echo "[$(date +%T)] attempt $try failed"
    sleep 20
done
[ "$ok" = "1" ] || { echo "[$(date +%T)] rpp failed 3 attempts -- genuinely broken, not transient"; exit 1; }

echo "[$(date +%T)] === baseline rpp (30 trials) ==="
mkdir -p evaluation/results/base_rpp
POSES_CSV="$PWD/evaluation/results/bigarena_poses.csv" \
BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
    ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 rpp \
    > evaluation/logs/base_rpp.log 2>&1
cp "$CSV" evaluation/results/base_rpp/batch.csv

ad=evaluation/bags/archive_base_rpp; mkdir -p "$ad"
for d in evaluation/bags/rpp__rpp_seed*; do
    case "$d" in *__prev_*) continue;; esac
    [ -f "$d/metadata.yaml" ] || continue
    cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
done
echo "[$(date +%T)] rpp done -> evaluation/results/base_rpp/batch.csv (bags $(ls "$ad" | wc -l))"
