#!/usr/bin/env bash
# Four stages closing the gaps that currently favour the baselines, 15 trials
# per pose set (A and B) = 30 per stage.
#
# Everything below is one variable at a time, each stage building on the last,
# so every stage is a paired comparison against the one before it on the same
# 30 routes:
#
#   2  vx_min -0.20 -> -0.35     the chassis is mecanum; reverse thrust is
#                                symmetric and vx_min looks inherited from a
#                                differential-drive default. It is also the
#                                binding constraint on every CBF encounter:
#                                net escape speed against the 0.10 m/s movers
#                                goes 0.10 -> 0.25 m/s.
#   3  replan 3.0 s -> 1.0 s     3 s dates from when the planner marked its
#                                costmap from the raw /scan and the path flipped
#                                every replan. It now reads /scan_filtered, so
#                                that reason is gone -- and nav2's own BT gives
#                                MPPI and RPP 1 Hz, three times our rate.
#   4  accel 0.8/0.6/1.2 -> 1.5/1.0/2.0
#                                Not a physical limit: gz VelocityControl has no
#                                actuator model, and OUR OWN velocity_smoother
#                                already permits 1.5/1.0/2.0 -- the same numbers
#                                MPPI ships with. The GMPC was self-limiting
#                                below its own chassis configuration.
#   5  baselines clamped down    BASE_ACCEL=1 puts MPPI and RPP on the GMPC's
#                                original box. Kept as a separate arm because a
#                                reviewer will ask from both directions: does
#                                the result hold when the baseline is given its
#                                author-recommended settings, and when it is
#                                held to ours.
#
# 30 trials per stage detects only large effects: the contact rate to move is
# 10/117 = 8.5%, whose 95% interval at n=30 spans roughly 3-23%. Stages that
# look flat here are NOT evidence of no effect.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
Y=src/ammr_wholebody_mpc/config/gmpc_params.yaml
R=evaluation/run_omnibot_dynamic.sh

echo "[$(date +%T)] waiting for the final benchmark ..."
while ! grep -qE "final benchmark complete|ABORTED" evaluation/logs/chainFinal.log 2>/dev/null; do sleep 120; done
while [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; do sleep 60; done
sleep 30

# Only safe between batches: bash reads this script line by line while it runs.
if ! grep -q REPLAN "$R"; then
    sed -i 's/ MIN_CLUSTER_PTS EKF_REJECT; do/ MIN_CLUSTER_PTS EKF_REJECT REPLAN BASE_ACCEL; do/' "$R"
    bash -n "$R" && grep -q REPLAN "$R" || { echo "FATAL: sed broke $R"; exit 1; }
    echo "[$(date +%T)] forwarding list: added REPLAN, BASE_ACCEL"
fi

edit_yaml () {                   # $1 key  $2 old  $3 new
    grep -q "^ *$1: *$2\$" "$Y" || return 0
    sed -i "s/^\( *$1: *\)$2\$/\1$3/" "$Y"
    grep -q "^ *$1: *$3\$" "$Y" || { echo "FATAL: $1 edit failed"; exit 1; }
    echo "[$(date +%T)] $1: $2 -> $3"
}
rebuild () {
    source /opt/ros/jazzy/setup.bash
    colcon build --packages-select ammr_wholebody_mpc > evaluation/logs/build_fair.log 2>&1
    for pair in "$@"; do
        grep -q "$pair" install/ammr_wholebody_mpc/share/ammr_wholebody_mpc/config/gmpc_params.yaml \
            || { echo "FATAL: install/ missing $pair"; exit 1; }
    done
    echo "[$(date +%T)] build verified"
}

run_one () {                     # $1 outdir  $2 poses  $3 method  $4.. env
    local out="$1" poses="$2" m="$3"; shift 3
    local CSV="evaluation/results/omnibot_dynamic_${m}.csv"
    local pat; case "$m" in gmpc_scan) pat='gmpc_cbf__scan_seed';; *) pat="${m}__${m}_seed";; esac
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    echo "[$(date +%T)] === $out ($m) ==="
    local envs=(POSES_CSV="$PWD/$poses" BIGARENA=1 TRAJ=bigarena_traffic GUI=0 "$@")
    [ "$m" = gmpc_scan ] && envs+=(PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0
                                   CBF_ALPHA=0.5 CBF_SAFE_MARGIN=0.60
                                   CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2)
    env "${envs[@]}" ./evaluation/run_omnibot_dynamic.sh 15 250 0 0 "$m" \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; return 1; }
    wait $bpid
    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/${pat}*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le 15 ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$CSV" "$ad/results.csv"
    echo "[$(date +%T)] $out done (bags $(ls -d "$ad"/*_seed* 2>/dev/null | wc -l))"
}

PA=evaluation/results/bigarena_poses.csv
PB=evaluation/results/bigarena_poses_b.csv

# --- stage 2: symmetric vx -------------------------------------------------
edit_yaml vx_min '\-0\.20' '-0.35'
rebuild -- '-0.35'
run_one s2_vx_A "$PA" gmpc_scan
run_one s2_vx_B "$PB" gmpc_scan

# --- stage 3: + 1 Hz replanning -------------------------------------------
run_one s3_replan_A "$PA" gmpc_scan REPLAN=1.0
run_one s3_replan_B "$PB" gmpc_scan REPLAN=1.0

# --- stage 4: + full acceleration -----------------------------------------
edit_yaml ax_max '0\.8' '1.5'
edit_yaml ay_max '0\.6' '1.0'
edit_yaml az_max '1\.2' '2.0'
rebuild '1.5' '1.0' '2.0'
run_one s4_accel_A "$PA" gmpc_scan REPLAN=1.0
run_one s4_accel_B "$PB" gmpc_scan REPLAN=1.0

# --- stage 5: baselines clamped to the GMPC's original box ----------------
run_one s5_mppi_A "$PA" mppi BASE_ACCEL=1
run_one s5_mppi_B "$PB" mppi BASE_ACCEL=1
run_one s5_rpp_A  "$PA" rpp  BASE_ACCEL=1
run_one s5_rpp_B  "$PB" rpp  BASE_ACCEL=1

echo "[$(date +%T)] fairness sweep complete"
