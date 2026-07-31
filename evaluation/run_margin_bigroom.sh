#!/usr/bin/env bash
# Did static_cbf_safe_margin 0.33 -> 0.38 break the 20 m room?
#
# 0.38 was adopted from arena evidence: it took shapes from 8 collisions to 3
# and stopgo from 2 to 0, at no cost there. But the arena's openings are 1.4 m
# wide while the big room has ~0.9 m passages, and the margin is a keep-out from
# a wall CELL: at 0.33 it leaves 94.8% of free cells navigable, at 0.38 only
# 89.5%. A watched run since the change covered 5.6 m in 56 s -- half nominal
# speed -- with min_h below the danger threshold 59% of the time, against
# 134 s for the whole 37 m route before it.
cd /home/howardchen/masterthesis
for M in 0.33 0.38; do
  TAG="bm_${M/./p}"
  [ -d "evaluation/bags/archive_$TAG" ] && { echo "SKIP $TAG"; continue; }
  echo "=== big room  static_margin=$M ==="
  rm -f /tmp/omnibot_dynamic.pid
  PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5 STATIC_MARGIN=$M \
    ./evaluation/run_omnibot_dynamic.sh 5 250 17 17 gmpc_scan
  cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "evaluation/results/final_$TAG.csv"
  mkdir -p "evaluation/bags/archive_$TAG"
  for i in $(seq 1 5); do
    [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
      cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" "evaluation/bags/archive_$TAG/"
  done
done
echo "=== BM DONE ==="
