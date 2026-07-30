#!/usr/bin/env bash
# Two 10-trial groups back to back, so the comparison is same-scenario and the
# code cannot change between them (building mid-batch has silently split a
# group's trials across code states before).
cd /home/howardchen/masterthesis
rm -f /tmp/omnibot_dynamic.pid
echo "=== GROUP A: detour only (baseline at dyn_obs_1=0.15) ==="
DETOUR=1 ./evaluation/run_detour_ab.sh ab_A
rm -f /tmp/omnibot_dynamic.pid
echo "=== GROUP B: + plan blend 1.0 s + yaw lookahead 0.7 m ==="
DETOUR=1 PLAN_BLEND=1.0 YAW_LOOKAHEAD=0.7 ./evaluation/run_detour_ab.sh ab_B
echo "=== BOTH DONE ==="
