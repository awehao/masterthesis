#!/usr/bin/env bash
# Does holding still when blocked fix the both-gaps livelock?
#
# Measured without it: 8 left/right plan flips in 29 replans, 10.9 reversals per
# run, 12.7% of commands in reverse, the robot never reached the divider in 88 s
# -- 2 of 10 trials arrived and 9 collided. The planner is decoupled from the
# movers, so it cannot know an opening is blocked and keeps routing through one;
# the CBF stops the robot; three seconds later the plan flips to the other
# opening. Nothing in the stack could express "wait".
#
# Also run on gapblock (one opening blocked, the other free) to check the
# detector does not fire when a route genuinely exists -- there it must stay
# out of the way, since that scenario already passes 9/9 with no collisions.
cd /home/howardchen/masterthesis
for S in bothgaps gapblock; do
  for W in 0 8; do
    TAG="stuck_${S}_w${W}"
    [ -d "evaluation/bags/archive_$TAG" ] && { echo "SKIP $TAG"; continue; }
    echo "=== $S  stuck_window=$W ==="
    rm -f /tmp/omnibot_dynamic.pid
    ARENA=1 TRAJ="arena_$S" PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 \
      CBF_ALPHA=1.5 STUCK_WINDOW=$W \
      ./evaluation/run_omnibot_dynamic.sh 10 120 6.6 6.6 gmpc_scan
    cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "evaluation/results/final_$TAG.csv"
    mkdir -p "evaluation/bags/archive_$TAG"
    for i in $(seq 1 10); do
      [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
        cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" "evaluation/bags/archive_$TAG/"
    done
  done
done
echo "=== STUCK AB DONE ==="
