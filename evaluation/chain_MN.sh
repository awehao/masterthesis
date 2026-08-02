#!/usr/bin/env bash
# Batches M then N, each guarded against the dead-robot failure mode.
#
# No `set -u` (ROS setup.bash reads unset variables and would abort the script)
# and no `set -e` (a failed batch must not stop the next one).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi

run_guarded () {          # $1 = script, $2 = log, $3 = label
    echo "[$(date +%T)] === $3 ==="
    "$1" > "$2" 2>&1 &
    local bpid=$!
    # The guard checks the first finished trial and kills the batch if the
    # robot never moved -- the failure that cost 18 trials on the last attempt.
    ./evaluation/guard_first_trial.sh || {
        echo "[$(date +%T)] $3 ABORTED by guard"; wait $bpid 2>/dev/null; return 1; }
    wait $bpid
    echo "[$(date +%T)] $3 exit $?"
}

run_guarded ./evaluation/run_noyaw.sh         evaluation/logs/noyawM.log   "batch M (no heading reference)" \
    && run_guarded ./evaluation/run_noyaw_noblend.sh evaluation/logs/noblendN.log "batch N (M + PLAN_BLEND=0)"

echo "[$(date +%T)] chain complete"
