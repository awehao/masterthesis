#!/usr/bin/env bash
# cbf_alpha sweep. Everything else is held at the validated detour config --
# in particular PLAN_BLEND and YAW_LOOKAHEAD stay off, because those are not
# validated yet and would confound the one variable being swept.
#
# alpha is the class-K gain in h_dot >= -alpha*h. Smaller engages the barrier
# earlier and more gently. At 3.0 the robot sits under the danger threshold
# ~48% of a run and saturates the acceleration box 21% of the time.
#
# Groups run back to back with no build in between: building mid-batch has
# already once split a group's trials across code states.
cd /home/howardchen/masterthesis
for A in 3.0 2.0 1.5; do
  rm -f /tmp/omnibot_dynamic.pid
  TAG="alpha${A/./p}"
  echo "=== GROUP alpha=$A -> $TAG ==="
  DETOUR=1 CBF_ALPHA="$A" ./evaluation/run_detour_ab.sh "$TAG"
done
echo "=== SWEEP DONE ==="
