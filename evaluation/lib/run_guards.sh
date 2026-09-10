# Shared guards for the benchmark runners.  source this file; it defines
# functions only and touches no global state.
#
# Each function here exists because its inline predecessor passed when it should
# have failed.  They are covered by evaluation/test_run_guards.sh, which runs
# them against stub `ros2` commands -- so a regression shows up in a test rather
# than in a trial that has to be thrown away.

# --- lifecycle -------------------------------------------------------------
# `ros2 lifecycle get X` prints e.g. "active [3]" or "inactive [2]".  The
# previous check was `grep -q active`, and "inactive" contains "active", so an
# unconfigured node passed the readiness gate.  Match the state word itself.
lifecycle_active() {
    local out state
    out=$(timeout 6 ros2 lifecycle get "$1" 2>/dev/null) || return 1
    state=$(printf '%s\n' "$out" | head -1 | awk '{print $1}')
    [ "$state" = "active" ]
}

# --- run a step without losing its exit code -------------------------------
# `cmd | tee -a "$LOG" | sed ...` makes the shell judge sed, so a failing python
# step reported success.  This runs the command with its output captured, then
# replays it into the log with a prefix, and returns the COMMAND's status.
#   run_step <logfile> <prefix> <outfile> <cmd...>
run_step() {
    local log="$1" prefix="$2" out="$3"; shift 3
    "$@" > "$out" 2>&1
    local rc=$?
    sed "s|^|${prefix}|" "$out" | tee -a "$log"
    return $rc
}

# --- method -> configuration ----------------------------------------------
# METHOD used to select only the directory name, while the heading objective
# came from a separate HEADING variable, so "gmpc_scan_heading" with HEADING=0
# was a legal, silent and completely wrong ON/OFF pair.  One table, and an
# unknown method is refused rather than quietly falling through to the default
# chain.  Prints shell assignments to eval; returns 1 if the method is unknown.
method_config() {
    case "$1" in
        gmpc_scan)          echo "HEADING=0" ;;
        gmpc_scan_heading)  echo "HEADING=1" ;;
        *) echo "unknown method '$1' (known: gmpc_scan, gmpc_scan_heading)" >&2
           return 1 ;;
    esac
}

# --- read the configuration back off the running node ----------------------
# The mapping above says what was INTENDED.  This says what the node actually
# came up with, which is the only version worth putting in a manifest.
#   readback_param <node> <param>   -> prints the value, or fails
readback_param() {
    local out
    out=$(timeout 10 ros2 param get "$1" "$2" 2>/dev/null) || return 1
    case "$out" in
        *"No parameter"*|"") return 1 ;;
    esac
    printf '%s\n' "${out##*: }"
}

# Cross-check one parameter against the value the method table implies.
#   assert_param <node> <param> <expected>
assert_param() {
    local got
    got=$(readback_param "$1" "$2") || {
        echo "!! 讀不到 $1 的參數 $2" >&2; return 1; }
    if [ "$got" != "$3" ]; then
        echo "!! $1 $2 = $got，但方法要求 $3" >&2
        return 1
    fi
    echo "   $1 $2 = $got（符合方法設定）"
}

# --- did the simulator finish, or just die? --------------------------------
# The runner printed "已結束並保存" as soon as the process vanished, whether it
# had saved a result or segfaulted.  Wait for it, keep its status, and check the
# result file actually holds the fields a trial is scored on.
#   verify_result_json <path> [required keys...]
verify_result_json() {
    local f="$1"; shift
    [ -s "$f" ] || { echo "!! 結果檔不存在或為空：$f" >&2; return 1; }
    python3 - "$f" "$@" <<'PYEOF'
import json, sys
path, keys = sys.argv[1], sys.argv[2:] or ['stop_reason', 'log', 'sim_time']
try:
    d = json.load(open(path))
except Exception as e:
    sys.exit(f'!! 結果檔無法解析：{e}')
run = d.get('run', d)
missing = [k for k in keys if run.get(k) is None]
if missing:
    sys.exit(f'!! 結果檔缺少必要欄位：{", ".join(missing)}')
if not run.get('log'):
    sys.exit('!! 結果檔沒有任何取樣，無法評分')
print(f'   結果檔完整：stop_reason={run["stop_reason"]}，'
      f'{len(run["log"])} 筆取樣，sim_time={run.get("sim_time")}')
PYEOF
}
