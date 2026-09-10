#!/usr/bin/env bash
# Regression tests for evaluation/lib/run_guards.sh.
#
# Every case below is a failure that previously passed.  They run against stub
# `ros2` / stub commands on PATH, so no simulator, no ROS graph, no run.
#
#   bash evaluation/test_run_guards.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/lib/run_guards.sh"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
STUB="$TMP/bin"; mkdir -p "$STUB"; PATH="$STUB:$PATH"
FAILS=0

check() {  # check <name> <expected 0|1> <actual rc>
    if [ "$2" -eq "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s   (expected rc=%s, got %s)\n' "$1" "$2" "$3"; FAILS=$((FAILS+1)); fi
}

stub_lifecycle() {  # write a fake `ros2` whose `lifecycle get` prints $1
    cat > "$STUB/ros2" <<EOF
#!/usr/bin/env bash
if [ "\$1" = "lifecycle" ]; then printf '%s\n' "$1"; exit 0; fi
exit 0
EOF
    chmod +x "$STUB/ros2"
}

echo
echo "-- lifecycle_active --"
stub_lifecycle 'active [3]'
lifecycle_active /amcl; check "'active [3]' is active" 0 $?
# The one that mattered: substring matching accepted a node that never configured.
stub_lifecycle 'inactive [2]'
lifecycle_active /amcl; check "'inactive [2]' is NOT active" 1 $?
stub_lifecycle 'unconfigured [1]'
lifecycle_active /amcl; check "'unconfigured [1]' is NOT active" 1 $?
stub_lifecycle 'finalized [4]'
lifecycle_active /amcl; check "'finalized [4]' is NOT active" 1 $?
cat > "$STUB/ros2" <<'EOF'
#!/usr/bin/env bash
exit 1
EOF
chmod +x "$STUB/ros2"
lifecycle_active /amcl; check "a failing lifecycle query is NOT active" 1 $?

echo
echo "-- run_step keeps the command's exit code --"
cat > "$STUB/failer" <<'EOF'
#!/usr/bin/env bash
echo "移動確認未通過"; exit 3
EOF
chmod +x "$STUB/failer"
LOG="$TMP/run.log"
# The old form was `cmd | tee -a "$LOG" | sed ...`, which reports sed's status.
"$STUB/failer" | tee -a "$LOG" | sed 's/^/  /' >/dev/null
check "the OLD pipeline really did swallow exit code 3" 0 $?
run_step "$LOG" "  [5/6]   " "$TMP/step.out" "$STUB/failer" >/dev/null
check "run_step propagates exit code 3" 3 $?
grep -q "移動確認未通過" "$LOG"; check "run_step still writes the output to the log" 0 $?

echo
echo "-- method_config --"
[ "$(method_config gmpc_scan)" = "HEADING=0" ]; check "gmpc_scan -> HEADING=0" 0 $?
[ "$(method_config gmpc_scan_heading)" = "HEADING=1" ]; check "gmpc_scan_heading -> HEADING=1" 0 $?
method_config gmpc_scan_headnig >/dev/null 2>&1; check "a typo'd method is refused, not defaulted" 1 $?
method_config '' >/dev/null 2>&1; check "an empty method is refused" 1 $?

echo
echo "-- assert_param --"
cat > "$STUB/ros2" <<'EOF'
#!/usr/bin/env bash
if [ "$1" = "param" ] && [ "$2" = "get" ]; then
  echo "Boolean value is: ${FAKE_PARAM:-False}"; exit 0
fi
exit 0
EOF
chmod +x "$STUB/ros2"
FAKE_PARAM=True  assert_param /gmpc_node heading_enable True  >/dev/null; check "matching parameter passes" 0 $?
# The ON/OFF pair this protects: directory says _heading, node says heading off.
FAKE_PARAM=False assert_param /gmpc_node heading_enable True  >/dev/null 2>&1; check "heading OFF under a _heading method is caught" 1 $?
FAKE_PARAM=True  assert_param /gmpc_node heading_enable False >/dev/null 2>&1; check "heading ON under a plain method is caught" 1 $?
cat > "$STUB/ros2" <<'EOF'
#!/usr/bin/env bash
echo "No parameter named 'heading_enable'"; exit 0
EOF
chmod +x "$STUB/ros2"
assert_param /gmpc_node heading_enable True >/dev/null 2>&1; check "a missing parameter fails instead of comparing empty" 1 $?

echo
echo "-- verify_result_json --"
echo '{"run": {"stop_reason": "goal_reached_truth", "log": [{"t": 1.0}], "sim_time": 12.0}}' > "$TMP/ok.json"
verify_result_json "$TMP/ok.json" >/dev/null; check "a complete result passes" 0 $?
echo '{"run": {"stop_reason": null, "log": [{"t": 1.0}], "sim_time": 12.0}}' > "$TMP/nostop.json"
verify_result_json "$TMP/nostop.json" >/dev/null 2>&1; check "stop_reason=null is caught (the truncated-save case)" 1 $?
echo '{"run": {"stop_reason": "duration", "log": [], "sim_time": 0.0}}' > "$TMP/nolog.json"
verify_result_json "$TMP/nolog.json" >/dev/null 2>&1; check "an unscoreable, sample-free result is caught" 1 $?
printf '{"run": {"stop_rea' > "$TMP/trunc.json"
verify_result_json "$TMP/trunc.json" >/dev/null 2>&1; check "a truncated JSON is caught, not crashed on" 1 $?
verify_result_json "$TMP/missing.json" >/dev/null 2>&1; check "a missing result file is caught" 1 $?

echo
if [ "$FAILS" -eq 0 ]; then echo "all run-guard shell tests pass"; else echo "FAILURES: $FAILS"; fi
exit $((FAILS > 0))
