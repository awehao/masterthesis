#!/usr/bin/env bash
# Per-trial provenance: what code, what actual parameters, which bag.
#
# Every number in the write-up has to survive three questions -- which build,
# which resolved parameters, which bags -- and until now none of them could be
# answered after the fact. That gap produced a "CBF acts at 0.87 m" figure no
# script could reproduce, a 28.1% split whose denominator had drifted, and a
# week of doubt over whether the movers ran at 0.14 or 0.18 m/s. This tool
# writes the answer down while the trial is running, when it is still cheap.
#
#   archive_experiment_manifest.sh prepare  <run_id> <method> <seed> <bag_dir>
#   archive_experiment_manifest.sh params   <run_id>
#   archive_experiment_manifest.sh finalize <run_id> <status> <exit_code>
#
# Output: evaluation/results/manifests/<run_id>/
#   manifest.json          the record, including SHA-256 of every input
#   environment.txt        whitelisted experiment variables only
#   ros_params.yaml        parameters DUMPED FROM THE LIVE NODES
#   git.diff               uncommitted changes, so a dirty tree is recoverable
#   inputs/                verbatim copies of the files the trial actually used
#
# Two principles worth stating because both were learned the hard way:
#
#   Archive the real input, not the intended one. Paths get edited between
#   batches; content hashes do not. generate_bigarena.py said 0.18 while the
#   YAML the batch loaded said 0.14, and nothing recorded which one ran.
#
#   Dump parameters from the running nodes, not from the YAML. A YAML and a set
#   of environment variables describe what was meant to be passed. Only
#   `ros2 param dump` shows what a node actually received.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/.." && pwd)"
ROOT="${HERE}/results/manifests"

# Whitelist, never the whole environment: a blanket dump would drag in tokens,
# proxies and home paths. Keep this list in step with the knobs the launch
# files actually read.
EXPERIMENT_ENV_KEYS=(
  METHOD SEED GUI ARM BIGARENA ARENA TRAJ SCENE POSES_CSV
  HORIZON PLAN_BLEND PLANNER_SCAN INFLATION REPLAN
  CBF_ALPHA CBF_SAFE_MARGIN STATIC_MARGIN MARGIN_MODE CBF_SLACK_W
  CBF_PRUNE_RANGE CBF_NEAR_STEPS CBF_FAR_STRIDE CBF_MARGIN_GROWTH
  CBF_VEL_MARGIN CBF_STATIC_SLACK_SCALE HARD_K0 HARD_K0_STATIC
  SHIELD SHIELD_ALPHA SHIELD_TAU
  MIN_TRACK_SPEED TRACK_RELEASE_SPEED TRACK_RELEASE_FRAMES
  MIN_NET_SPEED STATIC_WINDOW STATIC_KEEP_VEL MIN_CLUSTER_PTS
  ASSOC_PREDICT FRAG_MERGE COAST_S
  POSE_SOURCE EKF_REJECT MASK_HW
  AX_MAX AY_MAX AZ_MAX BASE_ACCEL
  ST_WEIGHT ST_SIGMA0 ST_GROWTH PROG_WEIGHT
  DETOUR DETOUR_OFFSET DETOUR_RANGE DETOUR_VX_FLOOR
  DETOUR_CLEAR_REF DETOUR_CLEAR_PAD DETOUR_SIDE_PROJ
  STUCK_WINDOW STUCK_PROGRESS STUCK_RELEASE_H
  GOAL_DELAY_MIN GOAL_DELAY_MAX SPAWN_X SPAWN_Y
)

sha() { [ -f "$1" ] && sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || echo ""; }
jstr() { python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "${1:-}"; }

# Copy an input file next to the manifest and echo "<name>|<sha>|<archived>".
# Copying matters: a path in a JSON file is not evidence once someone edits the
# file it points at.
keep_input() {
    local key="$1"
    local src="${2:-}"
    local dir="$3"
    [ -n "$src" ] && [ -f "$src" ] || { echo "${key}||"; return; }
    mkdir -p "$dir/inputs"
    local base; base="$(basename "$src")"
    cp -f "$src" "$dir/inputs/$base" 2>/dev/null || true
    echo "${key}|$(sha "$src")|inputs/${base}"
}

cmd_prepare() {
    local run_id="$1"
    local method="$2"
    local seed="$3"
    local bag_dir="${4:-}"
    local dir="${ROOT}/${run_id}"
    rm -rf "$dir"; mkdir -p "$dir/inputs"

    { for k in "${EXPERIMENT_ENV_KEYS[@]}"; do
        printf '%s=%s\n' "$k" "${!k-<unset>}"
      done; } > "$dir/environment.txt"

    # Uncommitted changes, excluding anything large or generated. The same
    # commit in a dirty tree can be a completely different program.
    ( cd "$WS" && git status --short > "$dir/git.status" 2>/dev/null
      # Source only. Bags, results and logs are outputs of the very run being
      # recorded, so including them makes the diff unapplicable (git refuses to
      # add a file the run has already written) and bloats it for no benefit.
      EXC=( ':(exclude)evaluation/bags' ':(exclude)evaluation/results'
            ':(exclude)evaluation/logs' ':(exclude)build' ':(exclude)install'
            ':(exclude)log' )
      git diff --binary -- . "${EXC[@]}" > "$dir/git.diff" 2>/dev/null
      git diff --cached --binary -- . "${EXC[@]}" > "$dir/git.staged.diff" 2>/dev/null ) || true

    local commit dirty
    commit="$(cd "$WS" && git rev-parse HEAD 2>/dev/null || echo unknown)"
    dirty=false; [ -s "$dir/git.status" ] && dirty=true

    # The actual inputs this trial reads.
    local world traj poses mapyaml mapimg gmpcyaml ctrlyaml
    local arena="${BIGARENA:-0}"
    if [ "$arena" = "1" ]; then world="$WS/src/ammr_bringup/worlds/bigarena.sdf"
    else world="$WS/src/ammr_bringup/worlds/random_room_dynamic.sdf"; fi
    traj="$WS/src/ammr_bringup/config/dynamic_trajectories_${TRAJ:-bigarena_traffic}.yaml"
    poses="${POSES_CSV:-}"
    mapyaml="$WS/src/ammr_bringup/maps/$([ "$arena" = 1 ] && echo bigarena || echo random_room).yaml"
    mapimg="$(python3 - "$mapyaml" <<'PY' 2>/dev/null || true
import sys,os,yaml
try:
    y=yaml.safe_load(open(sys.argv[1])); im=y.get('image','')
    print(im if os.path.isabs(im) else os.path.join(os.path.dirname(sys.argv[1]),im))
except Exception: pass
PY
)"
    gmpcyaml="$WS/src/ammr_wholebody_mpc/config/gmpc_params.yaml"
    case "${method}" in
      mppi) ctrlyaml="$WS/src/ammr_navigation/config/nav2_params_mppi.yaml" ;;
      rpp)  ctrlyaml="$WS/src/ammr_navigation/config/nav2_params_rpp.yaml"  ;;
      *)    ctrlyaml="$gmpcyaml" ;;
    esac

    local rows=()
    rows+=( "$(keep_input world       "$world"     "$dir")" )
    rows+=( "$(keep_input trajectory  "$traj"      "$dir")" )
    rows+=( "$(keep_input poses       "$poses"     "$dir")" )
    rows+=( "$(keep_input map_yaml    "$mapyaml"   "$dir")" )
    rows+=( "$(keep_input map_image   "$mapimg"    "$dir")" )
    rows+=( "$(keep_input gmpc_config "$gmpcyaml"  "$dir")" )
    rows+=( "$(keep_input controller_config "$ctrlyaml" "$dir")" )

    python3 - "$dir/manifest.json" "$run_id" "$method" "$seed" "$bag_dir" \
              "$commit" "$dirty" "${rows[@]}" <<'PY'
import json, os, sys, datetime
out, run_id, method, seed, bag, commit, dirty = sys.argv[1:8]
inputs = {}
for row in sys.argv[8:]:
    k, h, path = (row.split('|') + ['', ''])[:3]
    inputs[k] = {'sha256': h or None, 'archived_copy': path or None}
json.dump({
    'schema_version': 1,
    'run_id': run_id,
    'method': method,
    'seed': int(seed) if str(seed).isdigit() else seed,
    'status': 'starting',
    'start_time': datetime.datetime.now().astimezone().isoformat(),
    'git_commit': commit,
    'git_dirty': dirty == 'true',
    'ros_distro': os.environ.get('ROS_DISTRO'),
    'bag_path': bag or None,
    'environment_file': 'environment.txt',
    'resolved_parameters_file': None,
    'inputs': inputs,
}, open(out, 'w'), indent=2, ensure_ascii=False)
open(out, 'a').write('\n')
PY
    echo "  manifest -> ${dir}"
}

# Dump parameters from the LIVE nodes. A node that is not running is recorded
# as not_running rather than skipped: "the shield was on" has to be provable,
# and an absent section reads the same as an absent shield.
cmd_params() {
    # NOT one `local` statement: bash declares every name in a `local` list
    # before evaluating any of the assignments, so `local a=$1 b=${ROOT}/$a`
    # reads $a while it is still unset and trips `set -u`.
    local run_id="$1"
    local dir="${ROOT}/${run_id}"
    [ -d "$dir" ] || { echo "  no manifest for ${run_id}"; return 1; }
    local nodes=(/gmpc_controller /scan_obstacle_tracker /scan_safety_shield
                 /velocity_smoother /controller_server /obstacle_aggregator)
    : > "$dir/ros_params.yaml"
    local absent=()
    local failed=()

    # "the node was not running" and "the dump did not come back" are different
    # claims and must not collapse into one label. The first is a fact about
    # the experiment; the second is a fact about this script, and silently
    # reporting it as the first would misrepresent the run. So ask node list
    # first, then retry the dump -- gmpc_controller solves a QP at 20 Hz and
    # does not always service the parameter request on the first try, which is
    # exactly how the first smoke test lost the most important node.
    local live
    live="$(timeout 20 ros2 node list 2>/dev/null || true)"
    for n in "${nodes[@]}"; do
        if ! grep -qx -- "$n" <<< "$live"; then
            absent+=( "${n#/}" )
            printf '# %s: not_running\n\n' "${n#/}" >> "$dir/ros_params.yaml"
            continue
        fi
        local ok=0 try
        for try in 1 2 3; do
            if timeout 20 ros2 param dump "$n" >> "$dir/ros_params.yaml" 2>/dev/null; then
                echo "" >> "$dir/ros_params.yaml"; ok=1; break
            fi
            sleep 2
        done
        if [ "$ok" != "1" ]; then
            failed+=( "${n#/}" )
            printf '# %s: RUNNING but param dump failed after 3 tries\n\n' "${n#/}" \
                >> "$dir/ros_params.yaml"
        fi
    done
    python3 - "$dir/manifest.json" "${#absent[@]}" "${absent[@]:-}" "${failed[@]:-}" <<'PY2'
import json, sys
p = sys.argv[1]
na = int(sys.argv[2])
rest = [x for x in sys.argv[3:]]
absent, failed = rest[:na], [x for x in rest[na:] if x]
j = json.load(open(p))
j['resolved_parameters_file'] = 'ros_params.yaml'
j['nodes_not_running'] = [x for x in absent if x]
j['nodes_param_dump_failed'] = failed
json.dump(j, open(p, 'w'), indent=2, ensure_ascii=False)
open(p, 'a').write('\n')
PY2
    echo "  params -> ${dir}/ros_params.yaml${absent[*]:+ (not running: ${absent[*]})}${failed[*]:+ (DUMP FAILED: ${failed[*]})}"
}

cmd_finalize() {
    local run_id="$1"
    local status="${2:-completed}"
    local code="${3:-0}"
    local dir="${ROOT}/${run_id}"
    [ -d "$dir" ] || return 0
    python3 - "$dir/manifest.json" "$status" "$code" <<'PY'
import json, os, sys, datetime
p, status, code = sys.argv[1], sys.argv[2], sys.argv[3]
j = json.load(open(p))
bag = j.get('bag_path') or ''
j.update({
    'status': status,
    'end_time': datetime.datetime.now().astimezone().isoformat(),
    'exit_code': int(code) if str(code).lstrip('-').isdigit() else None,
    'metadata_present': bool(bag) and os.path.exists(os.path.join(bag, 'metadata.yaml')),
})
j['bag_complete'] = j['metadata_present'] and any(
    f.endswith('.mcap') for f in (os.listdir(bag) if bag and os.path.isdir(bag) else []))
json.dump(j, open(p, 'w'), indent=2, ensure_ascii=False)
open(p, 'a').write('\n')
print(f"  manifest {status} (bag_complete={j['bag_complete']})")
PY
}

case "${1:-}" in
    prepare)  shift; cmd_prepare  "$@" ;;
    params)   shift; cmd_params   "$@" ;;
    finalize) shift; cmd_finalize "$@" ;;
    *) echo "usage: $0 {prepare|params|finalize} ..." >&2; exit 2 ;;
esac
