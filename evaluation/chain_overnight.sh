#!/usr/bin/env bash
# Overnight run: get the existing scenario to zero contacts, and be able to say
# WHY it got there.
#
# Order matters and each step depends on the one before:
#   0  wait for the running A/B, then analyse it
#   1  rebuild ammr_bringup so the mover speed cap (0.15/0.16/0.18 -> 0.14)
#      takes effect. Those three were the only movers whose t_0.1 = 0.1/(v_esc
#      - v_obs) reached or exceeded the 1.0 s horizon, i.e. a head-on encounter
#      inside d* had no solution at all. The other seven are untouched, so the
#      scenario's density, closure and encounter rate do not change.
#      NOT done earlier on purpose: rebuilding mid-batch swaps the world under
#      the trials still to run.
#   2  arm C  hard k=0 for walls only, new mover cap, routes 1-40
#   3  arm D  as-is soft slack,        new mover cap, routes 1-40   (paired)
#   4  arm E  whichever of C/D is cleaner, routes 1-40 again -- a REPLICATE,
#      because trial-to-trial spread was measured at 0.22-0.37 m against a
#      0.11 m between-config effect (report S6.6), so one 40-trial arm cannot
#      tell a real difference from phase luck.
#
# Every arm is guarded: if trial 1 comes back with path_length = 0 the arm is
# abandoned rather than burning two hours on a broken build.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec 2>&1

LOCKFILE=/tmp/omnibot_dynamic.pid
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
SUM=evaluation/results/OVERNIGHT_SUMMARY.md
N=40

say () { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 0. let the A/B finish -------------------------------------------------
say "waiting for the A/B arms ..."
while ! grep -q "A/B complete" evaluation/logs/chainAB.log 2>/dev/null; do
    ps -eo cmd --no-headers | grep -q "[c]hain_hardstatic" || break
    sleep 60
done
while [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; do
    sleep 30
done
sleep 20
say "A/B done"

# ---- 1. rebuild with the capped mover speeds -------------------------------
say "rebuilding ammr_bringup (mover cap 0.14)"
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ammr_bringup 2>&1 | tail -2
grep -c "0.14" install/ammr_bringup/share/ammr_bringup/config/dynamic_trajectories_bigarena_traffic.yaml \
    | sed 's/^/  movers now at 0.14: /'

# ---- arms ------------------------------------------------------------------
run_arm () {                       # $1 outdir   $2.. extra env
    local out="$1"; shift
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    rm -f "$CSV"
    say "=== $out ==="
    env "$@" MASK_HW=10.0 POSES_CSV="$PWD/evaluation/results/bigarena_poses_big.csv" \
        BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh "$N" 250 0 0 gmpc_scan \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; say "$out ABORTED"; return 1; }
    wait $bpid
    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/gmpc_cbf__scan_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        local s; s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le "$N" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$CSV" "$ad/results.csv"
    local neg; neg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
                              $k+0 < 0 {n++} END{print n+0}' "$CSV")
    say "$out done: $(( $(wc -l < "$CSV") - 1 )) trials, $neg negative"
}

run_arm hardC HARD_K0_STATIC=1
run_arm softD HARD_K0_STATIC=0

# ---- 4. replicate the cleaner arm ------------------------------------------
pick=hardC
cneg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
                $k+0<0{n++} END{print n+0}' evaluation/results/hardC/batch.csv 2>/dev/null || echo 99)
dneg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
                $k+0<0{n++} END{print n+0}' evaluation/results/softD/batch.csv 2>/dev/null || echo 99)
[ "${dneg:-99}" -lt "${cneg:-99}" ] && pick=softD
say "replicating $pick (hardC $cneg neg, softD $dneg neg)"
if [ "$pick" = hardC ]; then run_arm repE HARD_K0_STATIC=1; else run_arm repE HARD_K0_STATIC=0; fi

# ---- 5. summarise ----------------------------------------------------------
say "writing summary"
python3 evaluation/summarise_overnight.py > "$SUM" 2>&1
say "ALL DONE -> $SUM"
