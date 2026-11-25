"""Expand a proposed experiment budget; never launch ROS or a simulator.

Core and comprehensive are alternative budgets. Shared run_key values must be
deduplicated if the core plan is extended. Asset/configuration fields remain
unresolved until pilot tuning and a version freeze have been completed.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math

OUT = Path(__file__).resolve().parent
TASKS = ["cylinder_pick_store", "marker_pick_store", "door_open_close", "drawer_open_close"]
LOCAL_METHODS = ["wb_mpc_full", "holistic_qp_adapted", "strict_park_wb_mpc"]
NAV_METHODS = ["nav_full", "mppi_omni_shared_guard", "dwb_omni_shared_guard", "rpp_shared_guard"]
NAV_ABLATIONS = ["gmpc_bare", "gmpc_pred_cbf", "gmpc_guard_only",
                 "gmpc_pred_cbf_guard", "gmpc_current_cbf_guard", "gmpc_pred_cbf_guard_speed"]
WB_ABLATIONS = ["direct_rate_model", "no_queued_command_prediction", "fixed_bias",
                "no_tcp_integral", "fixed_bias_no_tcp_integral",
                "horizon_alternative_1", "horizon_alternative_2", "horizon_alternative_3"]
MODULES = {
    "M0": ("執行模型資料蒐集", 24, 72),
    "N0": ("底盤軌跡追蹤", 0, 360),
    "N1": ("導航外部基線比較", 240, 960),
    "N2": ("導航模組消融：新增執行", 120, 480),
    "W0": ("自由空間協同追蹤", 0, 160),
    "W1": ("局部操作外部基線比較", 240, 720),
    "W2": ("全身控制消融：新增執行", 160, 640),
    "V1": ("真值操作目標診斷：新增執行", 0, 80),
    "C1": ("固定相對位姿接觸對照：新增執行", 0, 80),
    "R0": ("受控擾動：新增執行", 0, 320),
    "S1": ("選位策略對照：新增執行", 0, 80),
    "K1": ("門扇位姿容許範圍消融：新增執行", 0, 20),
    "F0": ("故障注入與實際停止", 45, 90),
    "E0": ("住宅端到端任務", 120, 240),
}


def expand(tier):
    comprehensive = tier == "comprehensive"
    n = 20 if comprehensive else 10
    rows = []
    used = set()

    def add(module, case, method, split="test", **kw):
        key = f"{module}/{case}/{method}"
        assert key not in used, key
        used.add(key)
        rows.append(dict(run_key=key, module=module, case_id=case, method=method,
                         split=split, paired_block=case, status="planned",
                         asset_id="TBD_before_freeze", configuration_sha256="TBD_before_freeze",
                         simulator="Isaac_Sim" if module not in ("N0", "N1", "N2") else "Gazebo_Harmonic",
                         **kw))

    # No core test episode becomes a training episode in the extended plan.
    roles = {1: "train", 2: "validation", 3: "test", 4: "test"}
    if comprehensive:
        roles.update({5: "train", 6: "train", 7: "validation", 8: "validation",
                      9: "test", 10: "test", 11: "test", 12: "test"})
    for load in ("empty", "nominal_object"):
        for region in ("folded", "middle", "extended"):
            for seed, split in roles.items():
                add("M0", f"M/{load}/{region}/excitation{seed:02}", "excitation_recording", split,
                    load=load, configuration_region=region, excitation_seed=seed)

    if comprehensive:
        for path in ("circle", "figure_eight", "s_curve", "lateral_and_yaw"):
            for initial in ("small", "medium", "large"):
                for seed in range(1, 11):
                    for method in ("gmpc_tracking", "wrapped_euclidean_mpc", "body_frame_pid"):
                        add("N0", f"NT/{path}/{initial}/seed{seed:02}", method,
                            path_family=path, initial_error_level=initial, case_seed=seed)

    layouts = ["open", "narrow", "cluttered"] + (["heldout_house"] if comprehensive else [])
    traffic = ["static", "crossing"] + (["multi_motion"] if comprehensive else [])
    for layout in layouts:
        for motion in traffic:
            for seed in range(1, n + 1):
                case = f"N/{layout}/{motion}/seed{seed:02}"
                for method in NAV_METHODS:
                    add("N1", case, method, layout=layout, traffic=motion, case_seed=seed)

    selected_nav = [(l, m) for l in ("open", "narrow")
                    for m in (["crossing", "multi_motion"] if comprehensive else ["crossing"])]
    references = []
    for layout, motion in selected_nav:
        for seed in range(1, n + 1):
            case = f"N/{layout}/{motion}/seed{seed:02}"
            references.append(dict(analysis="N2", run_key=f"N1/{case}/nav_full"))
            for method in NAV_ABLATIONS:
                add("N2", case, method, layout=layout, traffic=motion, case_seed=seed)

    if comprehensive:
        for move in ("base_translation_tcp_hold", "base_yaw_tcp_hold", "world_tcp_path", "base_relative_carry"):
            for seed in range(1, n + 1):
                for method in ("wb_mpc_full", "holistic_qp_adapted"):
                    add("W0", f"WF/{move}/seed{seed:02}", method, case_seed=seed, reference_family=move)

    conditions = ["nominal", "cluttered"] + (["dynamic_crossing"] if comprehensive else [])
    for task in TASKS:
        for condition in conditions:
            for seed in range(1, n + 1):
                case = f"W/{task}/{condition}/seed{seed:02}"
                for method in LOCAL_METHODS:
                    add("W1", case, method, task=task, condition=condition, case_seed=seed,
                        contact_model="physical_friction", target_source="rgbd_estimate")

    ablation_tasks = TASKS if comprehensive else ["cylinder_pick_store", "drawer_open_close"]
    for task in ablation_tasks:
        for seed in range(1, n + 1):
            case = f"W/{task}/nominal/seed{seed:02}"
            references.append(dict(analysis="W2", run_key=f"W1/{case}/wb_mpc_full"))
            for method in WB_ABLATIONS:
                add("W2", case, method, task=task, condition="nominal", case_seed=seed,
                    contact_model="physical_friction", target_source="rgbd_estimate")

    if comprehensive:
        for task in TASKS:
            for seed in range(1, n + 1):
                case = f"W/{task}/nominal/seed{seed:02}"
                for analysis in ("V1", "C1", "R0"):
                    references.append(dict(analysis=analysis, run_key=f"W1/{case}/wb_mpc_full"))
                add("V1", case, "wb_mpc_true_target", task=task, condition="nominal", case_seed=seed,
                    contact_model="physical_friction", target_source="simulator_truth_diagnostic")
                add("C1", case, "wb_mpc_fixed_relative_contact", task=task, condition="nominal", case_seed=seed,
                    contact_model="fixed_relative_constraint", target_source="rgbd_estimate")
                for disturbance in ("command_delay_160ms", "target_bias_20mm", "vision_dropout_500ms", "drive_gain_scale_080"):
                    add("R0", case, f"wb_mpc_{disturbance}", task=task, condition=disturbance,
                        case_seed=seed, contact_model="physical_friction", target_source="rgbd_estimate")
                references.append(dict(analysis="S1", run_key=f"W1/{case}/wb_mpc_full"))
                add("S1", case, "wb_mpc_nearest_feasible_base", task=task, condition="nominal",
                    case_seed=seed, contact_model="physical_friction", target_source="rgbd_estimate")
                if task == "door_open_close":
                    references.append(dict(analysis="K1", run_key=f"W1/{case}/wb_mpc_full"))
                    add("K1", case, "wb_mpc_no_pose_tube", task=task, condition="nominal",
                        case_seed=seed, contact_model="physical_friction", target_source="rgbd_estimate")

    for phase in ("navigation", "approach", "drawer_pull"):
        for fault in ("worker_timeout", "qp_infeasible", "perception_stale"):
            for seed in range(1, (10 if comprehensive else 5) + 1):
                add("F0", f"F/{phase}/{fault}/seed{seed:02}", "wb_mpc_fault_stop", phase=phase,
                    fault=fault, case_seed=seed)

    for layout in ("house_A", "house_B_heldout"):
        for motion in ("no_walkers", "scripted_walkers"):
            for seed in range(1, n + 1):
                for method in ("wb_mpc_full", "holistic_qp_adapted", "sequential_approach_wb_mpc"):
                    add("E0", f"E/{layout}/{motion}/seed{seed:02}", method,
                        layout=layout, traffic=motion, case_seed=seed,
                        contact_model="physical_friction", target_source="rgbd_estimate")

    counted = Counter(r["module"] for r in rows)
    for module, (_, core, full) in MODULES.items():
        assert counted[module] == (full if comprehensive else core), (module, counted[module])
    assert all(ref["run_key"] in used for ref in references)
    # Pair all variants of the same case into one scheduling block. This also
    # interleaves the reference method reused by an ablation with its variants.
    blocks = {}
    for row in rows:
        blocks.setdefault(row["paired_block"], []).append(row)
    for case, group in blocks.items():
        ordered = sorted(group, key=lambda r: hashlib.sha256((case+"|"+r["run_key"]).encode()).digest())
        for rank, row in enumerate(ordered, 1):
            row["within_case_order"] = rank
    return rows, references


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    result = {"scope": "simulation_only", "status": "proposal_not_executed", "budgets": {}}
    expanded = {}
    for tier in ("core", "comprehensive"):
        rows, refs = expand(tier)
        expanded[tier] = {r["run_key"]: r for r in rows}
        write_csv(OUT / f"{tier}_planned_runs.csv", rows)
        write_csv(OUT / f"{tier}_analysis_reuse.csv", refs)
        result["budgets"][tier] = dict(executions=len(rows), by_module=dict(Counter(r["module"] for r in rows)),
                                       reused_analysis_rows=len(refs),
                                       offline_test_scenes=60 if tier=="comprehensive" else 30,
                                       offline_test_images=600 if tier=="comprehensive" else 300)
    assert set(expanded["core"]) <= set(expanded["comprehensive"])
    assert all(expanded["core"][k]["split"]==expanded["comprehensive"][k]["split"] for k in expanded["core"])
    result["core_is_subset_of_comprehensive"] = True
    result["when_extending"] = "Deduplicate shared run_key values; keep the same frozen version, assets, thresholds and train/test split. Configuration changes require a new cohort."
    result["pilot_and_tuning_runs_included"] = False
    result["pilot_budget_hint"] = "30-60 development runs, kept separate from the formal test seed list"
    result["zero_failure_one_sided_95_upper_pct"] = {str(n):100*(1-.05**(1/n)) for n in [3,10,20,30,60,100]}
    result["zero_failure_trials_for_95pct_success_lower_bound"] = math.ceil(math.log(.05)/math.log(.95))
    result["paired_time_power_example"] = {"approximate_only":True,"paired_log_sd":.15,
        "relative_effect":.05,"z_normal_two_sided_95":1.95996398454,"z_power_80":.84162123357,
        "n_pairs_normal_approx":math.ceil(((1.95996398454+.84162123357)*.15/abs(math.log(.95)))**2)}
    result["files"] = [{"path": p.name, "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(OUT.glob("*.csv"))]
    (OUT / "budget_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result["budgets"],ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
