#!/usr/bin/env bash
# Does the sideways reference projection cost safety?
#
# It was added to stop the radial projection pulling the reference BACKWARDS
# when an obstacle sits dead ahead (measured 0.235 m of backward shift, and the
# robot answered by retreating 1.8 m down the spawn corridor). It removed that
# loop, but both ten-trial groups that carried it collided significantly more
# than the N=35 reference (4/10 p=0.006 and 3/10 p=0.030, Fisher exact), while
# the radial version did not (2/10, p=0.119).
#
# dyn_obs_1 is at 0.15 m/s, matching the N=35 (98%, 1/35) reference run, so the
# collision rates are directly comparable to it.
cd /home/howardchen/masterthesis
for SP in 0 1; do
  rm -f /tmp/omnibot_dynamic.pid
  echo "=== GROUP side_proj=$SP ==="
  DETOUR_SIDE_PROJ=$SP ./evaluation/run_detour_ab.sh "sp$SP"
done
echo "=== SIDEPROJ AB DONE ==="
