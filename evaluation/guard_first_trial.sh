#!/usr/bin/env bash
# Watch a running batch and kill it if its FIRST trial produced a dead robot.
#
# Batch M once ran 18 trials with path_length = 0.0 on every one: the controller
# raised AttributeError at startup and never published a command. Nothing in the
# harness noticed -- the launch came up, the recorder recorded, the analyser
# wrote rows. 90 minutes for nothing.
#
# A trial that ends with zero path length is never a legitimate result, so it is
# a safe abort condition. Checked once, on the first row, then this exits.
#
# usage: guard_first_trial.sh <results_csv>
CSV="${1:-evaluation/results/omnibot_dynamic_gmpc_scan.csv}"
LOCKFILE=/tmp/omnibot_dynamic.pid

# Wait for the first row, or for the batch to disappear.
while :; do
    [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null || exit 0
    [ -s "$CSV" ] && [ "$(( $(wc -l < "$CSV") - 1 ))" -ge 1 ] && break
    sleep 20
done

path=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="path_length_m") c=i; next}
                NR==2{print $c; exit}' "$CSV")
ok=$(awk -v p="$path" 'BEGIN{print (p+0 > 0.5) ? "1" : "0"}')
if [ "$ok" = "1" ]; then
    echo "guard: first trial moved ${path} m -- batch looks alive"
    exit 0
fi

echo "GUARD ABORT: first trial path_length=${path} -- the robot never moved."
echo "             Killing the batch instead of burning the night on it."
kill -TERM "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null
sleep 10
pkill -f "evaluation/run_omnibot_dynamic.sh" 2>/dev/null
exit 1
