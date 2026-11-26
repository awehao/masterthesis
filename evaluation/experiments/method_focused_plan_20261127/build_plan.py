"""Write a method-focused experiment proposal. Never launch experiments.

Counts refer to proposed executions, not existing results. Nominal full-method
runs are reused only when cases, targets, configuration and code are identical.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parent
TASKS = ("cylinder_pick_store", "marker_pick_store", "door_open_close", "drawer_open_close")
WB_METHODS = ("wb_mpc_full", "holistic_qp_adapted", "stop_before_approach_wb_mpc")
NAV_METHODS = ("nav_full", "mppi_omni_shared_guard", "dwb_omni_shared_guard", "rpp_shared_guard")
NAV_ABLATIONS = ("nav_no_motion_prediction", "nav_no_cbf", "nav_no_reference_adjustment", "nav_speed_only")
WB_ABLATIONS = ("wb_fixed_bias", "wb_no_tcp_integral", "wb_fixed_bias_no_tcp_integral",
                "wb_direct_rate_fixed_model_no_tcp_integral", "wb_horizon_1")
EXPECTED_CORE = {"M0": 12, "N1": 240, "N2": 80, "W1": 240, "W2": 100, "E0": 120}
EXPECTED_OPTIONAL = {"S1": 40, "K1": 10, "H1": 40}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, references, contrasts = [], [], []
    used = set()

    def add(module, case, method, tier="core", split="test", **kwargs):
        key = f"{module}/{case}/{method}"
        assert key not in used, key
        used.add(key)
        rows.append(dict(run_key=key, module=module, case_id=case, paired_block=case,
                         method=method, tier=tier, split=split, status="planned_not_executed",
                         configuration_sha256="TBD_before_test_freeze",
                         asset_id="TBD_before_test_freeze", **kwargs))
        return key

    def reuse(analysis, key):
        references.append(dict(analysis=analysis, run_key=key, additional_executions=0))

    def contrast(analysis, case, question, reference, variant, interpretation):
        contrasts.append(dict(analysis=analysis, case_id=case, question=question,
                              reference_run_key=reference, variant_run_key=variant,
                              interpretation=interpretation))

    # Nominal execution-model calibration: whole-record splits, no load sweep.
    for region in ("folded", "middle", "extended"):
        for seed, split in ((1, "train"), (2, "validation"), (3, "test"), (4, "test")):
            add("M0", f"M/{region}/recording{seed:02}", "nominal_excitation", split=split,
                configuration_region=region, recording_seed=seed,
                load="fixed_nominal_configuration")

    for layout in ("open", "narrow", "cluttered"):
        for traffic in ("static", "crossing"):
            for seed in range(1, 11):
                case = f"N/{layout}/{traffic}/seed{seed:02}"
                for method in NAV_METHODS:
                    add("N1", case, method, layout=layout, traffic=traffic, case_seed=seed,
                        raw_scan_guard="on_shared", norm_speed_and_wheel_limits="shared")

    for layout in ("open", "narrow"):
        for seed in range(1, 11):
            case = f"N/{layout}/crossing/seed{seed:02}"
            full = f"N1/{case}/nav_full"
            reuse("N2", full)
            variants = {}
            for method in NAV_ABLATIONS:
                variants[method] = add("N2", case, method, layout=layout, traffic="crossing",
                                       case_seed=seed, raw_scan_guard="on_shared",
                                       norm_speed_and_wheel_limits="shared")
            for question, method, interpretation in (
                ("obstacle_motion_prediction", "nav_no_motion_prediction",
                 "Zero obstacle velocity in BOTH candidate prediction and CBF; retain tracker, IDs and current geometry."),
                ("cbf_constraint", "nav_no_cbf",
                 "Remove only CBF rows and their unused slack costs; retain full reference adjustment and shared guard."),
                ("reference_adjustment_total", "nav_no_reference_adjustment",
                 "Remove speed and lateral reference adjustment; retain motion prediction in CBF and shared guard."),
                ("lateral_candidates", "nav_speed_only",
                 "Keep speed candidates, predictive CBF and shared guard; disable lateral candidates."),
            ):
                contrast("N2", case, question, full, variants[method], interpretation)
            contrast("N2", case, "speed_candidates", variants["nav_no_reference_adjustment"],
                     variants["nav_speed_only"],
                     "Only the speed-reference candidate layer differs between these two configurations.")

    for task in TASKS:
        for condition in ("nominal", "cluttered"):
            for seed in range(1, 11):
                case = f"W/{task}/{condition}/seed{seed:02}"
                for method in WB_METHODS:
                    add("W1", case, method, task=task, condition=condition, case_seed=seed,
                        target_source="shared_initial_rgbd_geometry_packet",
                        contact_model="shared_frozen_configuration",
                        nominal_horizon=("not_applicable_reactive_QP" if method == "holistic_qp_adapted"
                                         else "N_star_selected_on_validation"))

    for task in ("cylinder_pick_store", "drawer_open_close"):
        for seed in range(1, 11):
            case = f"W/{task}/nominal/seed{seed:02}"
            full = f"W1/{case}/wb_mpc_full"
            reuse("W2", full)
            variants = {}
            for method in WB_ABLATIONS:
                variants[method] = add("W2", case, method, task=task, condition="nominal",
                                       case_seed=seed, target_source="shared_initial_rgbd_geometry_packet",
                                       contact_model="shared_frozen_configuration")
            contrast("W2", case, "execution_tracking_model",
                     variants["wb_fixed_bias_no_tcp_integral"],
                     variants["wb_direct_rate_fixed_model_no_tcp_integral"],
                     "Both disable online bias adaptation and TCP integral; compare frozen identified augmented tracking model with direct rate integration.")
            contrast("W2", case, "prediction_horizon", full, variants["wb_horizon_1"],
                     "Same augmented model, dt, weights, terminal matrix, references, limits and online compensation; N_star versus N=1. Horizon changes cost extent and terminal time.")
            contrast("W2", case, "bias_with_integral", full, variants["wb_fixed_bias"],
                     "Freeze bias at its identified initial value; keep TCP integral enabled with unchanged gates.")
            contrast("W2", case, "integral_with_bias", full, variants["wb_no_tcp_integral"],
                     "Disable only TCP integral; keep causal online bias estimation and its gates.")
            contrast("W2", case, "bias_without_integral", variants["wb_no_tcp_integral"],
                     variants["wb_fixed_bias_no_tcp_integral"],
                     "Second bias contrast in the 2x2 bias-by-integral design.")
            contrast("W2", case, "integral_without_bias", variants["wb_fixed_bias"],
                     variants["wb_fixed_bias_no_tcp_integral"],
                     "Second integral contrast in the 2x2 bias-by-integral design.")
            for ordinal in (1, 2):
                alt = add("H1", case, f"wb_other_multistep_horizon_{ordinal}", tier="optional",
                          task=task, condition="nominal", case_seed=seed,
                          target_source="shared_initial_rgbd_geometry_packet",
                          contact_model="shared_frozen_configuration",
                          horizon="One of {5,10,20} other than N_star; resolve and freeze before test")
                contrast("H1", case, f"horizon_sensitivity_{ordinal}", full, alt,
                         "Optional: complete the N in {1,5,10,20} sensitivity curve; N=1 is already in W2.")
            reuse("H1", full)
            reuse("H1", variants["wb_horizon_1"])

    for task in TASKS:
        for seed in range(1, 11):
            case = f"W/{task}/nominal/seed{seed:02}"
            full = f"W1/{case}/wb_mpc_full"
            reuse("S1", full)
            alt = add("S1", case, "wb_nearest_feasible_base", tier="optional", task=task,
                      condition="nominal", case_seed=seed,
                      target_source="shared_initial_rgbd_geometry_packet",
                      contact_model="shared_frozen_configuration")
            contrast("S1", case, "operation_base_selection", full, alt,
                     "Keep candidate generation, route/IK/clearance feasibility and controller fixed; replace ranking only with nearest feasible base.")
            if task == "door_open_close":
                reuse("K1", full)
                alt = add("K1", case, "wb_no_door_pose_tube", tier="optional", task=task,
                          condition="nominal", case_seed=seed,
                          target_source="shared_initial_rgbd_geometry_packet",
                          contact_model="shared_frozen_configuration")
                contrast("K1", case, "door_pose_tube", full, alt,
                         "Remove only the hard door pose-tube rows; retain mechanism references, contact model, weights and other constraints.")

    for layout in ("house_A", "house_B_heldout"):
        for traffic in ("no_walkers", "scripted_walkers"):
            for seed in range(1, 11):
                case = f"E/{layout}/{traffic}/seed{seed:02}"
                for method in WB_METHODS:
                    add("E0", case, method, layout=layout, traffic=traffic, case_seed=seed,
                        target_source="online_rgbd", contact_model="shared_frozen_configuration")

    assert len(used) == len(rows)
    assert all(ref["run_key"] in used for ref in references)
    assert all(c[field] in used for c in contrasts for field in ("reference_run_key", "variant_run_key"))
    for tier, expected in (("core", EXPECTED_CORE), ("optional", EXPECTED_OPTIONAL)):
        assert dict(Counter(r["module"] for r in rows if r["tier"] == tier)) == expected
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["paired_block"]].append(row)
    for case, group in blocks.items():
        ordered = sorted(group, key=lambda r: hashlib.sha256((case + "|" + r["run_key"]).encode()).digest())
        for rank, row in enumerate(ordered, 1):
            row["within_case_order"] = rank

    core = [r for r in rows if r["tier"] == "core"]
    optional = [r for r in rows if r["tier"] == "optional"]
    write_csv(OUT / "core_planned_runs.csv", core)
    write_csv(OUT / "optional_planned_runs.csv", optional)
    write_csv(OUT / "analysis_reuse.csv", references)
    write_csv(OUT / "paired_contrasts.csv", contrasts)
    summary = dict(status="proposal_not_executed", date="2026-11-27", scope="simulation_only_method_focused",
                   core_executions=len(core), core_task_test_executions=len(core) - EXPECTED_CORE["M0"],
                   nominal_model_collection_executions=EXPECTED_CORE["M0"],
                   model_collection_split=dict(Counter(r["split"] for r in core if r["module"] == "M0")),
                   core_new_ablation_executions=EXPECTED_CORE["N2"] + EXPECTED_CORE["W2"],
                   core_by_module=EXPECTED_CORE, optional_by_module=EXPECTED_OPTIONAL,
                   optional_executions=len(optional), all_options_included_executions=len(rows),
                   reused_analysis_rows=len(references), paired_contrast_rows=len(contrasts),
                   pilot_tuning_or_vision_training_included=False,
                   warning="A seed denotes a distinct saved case, not repeated identical playback. Counts are workload proposals, not power guarantees.",
                   no_simulator_or_ros_launched=True)
    for name, value in (("budget_summary.json", summary),):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
