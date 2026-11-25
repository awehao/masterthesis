"""Chapter 4.6: archived timing, unsuccessful runs, and stopping evidence.

Reads saved files only. Never imports or starts the controller or simulator.
"""
from pathlib import Path
import hashlib
import json
import re

from build_execution_model_figures import ROOT, OUT, FONT, cn_label, legend, export
import matplotlib.pyplot as plt
import numpy as np

RUN = ROOT / "evaluation/runs/m8house_d2_unified_173"
COHORT = OUT / "residential_task_validation_data.json"
SOURCES = [COHORT, RUN / "d2_d2w.npz", RUN / "d2_d2w.json",
           RUN / "result.json", RUN / "run.log",
           ROOT / "evaluation/door_ahead_exec.py",
           ROOT / "evaluation/door_open_exec_ctrl.py",
           ROOT / "evaluation/isaac_m8_house_d2.py",
           ROOT / "evaluation/motm_dry/world_d2_node.py",
           ROOT / "evaluation/far_chain_align.py"]
MODES = [("grasp", "初段導航與抓持", "-"), ("open", "開門", "--"),
         ("pass", "穿門及導航", ":"), ("pick", "取放與回程", "-.")]


def read(path):
    return json.loads(path.read_text())


def percentile(values):
    a = np.asarray(values, float)
    assert a.size and np.isfinite(a).all()
    return {"n": int(a.size), "p50": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def timing():
    archive = np.load(RUN / "d2_d2w.npz", allow_pickle=False)
    logs = [json.loads(x) for x in archive["log"]]
    rows = [r for r in logs if r.get("mode") in {m[0] for m in MODES} and
            r.get("roundtrip_ms") is not None]
    assert len(rows) == 6688
    saved = read(RUN / "d2_d2w.json")
    simulator = read(RUN / "result.json")
    runlog = (RUN / "run.log").read_text()
    budgets = re.search(r"展示期限 ([0-9.]+)／([0-9.]+) ms", runlog)
    assert budgets and float(budgets.group(2)) == 120
    stats = {}
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4))
    fig.subplots_adjust(left=.14, right=.97, bottom=.11, top=.94, hspace=.43)
    ax = axes[0]
    for mode, label, line in MODES:
        group = [r for r in rows if r["mode"] == mode]
        for r in group:
            total = r["comm_in_ms"] + r["compute_ms"] + r["comm_out_ms"]
            assert abs(total - r["roundtrip_ms"]) <= .02  # rounded source fields
        stats[mode] = {"label": label, "roundtrip_ms": percentile([r["roundtrip_ms"] for r in group]),
                       "compute_ms": percentile([r["compute_ms"] for r in group]),
                       "n_above_50ms": sum(r["roundtrip_ms"] > 50 for r in group),
                       "n_above_120ms": sum(r["roundtrip_ms"] > 120 for r in group),
                       "n_committed": sum(bool(r.get("commit")) for r in group)}
        if mode in ("grasp", "open"):
            assert stats[mode]["roundtrip_ms"]["n"] == saved["by_mode"][mode]["n"]
            assert abs(stats[mode]["roundtrip_ms"]["p95"] - saved["by_mode"][mode]["roundtrip_ms"]["p95"]) < .1
        x = np.sort([r["roundtrip_ms"] for r in group])
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color="black", lw=1.1,
                ls=line, label=label)
    ax.axvline(50, color=".55", lw=.8)
    ax.text(51, .55, "50 ms", fontsize=10)
    cn_label(ax, x="派工至回覆之主機時間（ms）", y="累積比例", title="(a) 各控制區段之來回耗時分布")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 1.04)
    legend(ax, loc="lower right", ncol=1)
    opened = [r for r in rows if r["mode"] == "open"]
    breakdown = {key: percentile([r["parts"][key] for r in opened]) for key in
                 ("solve_ms", "cand_ms", "mon_ms", "tick_ms")}
    qp = [q for r in opened for q in r["parts"]["qp_all"]]
    assert len(qp) == 189 and all(q[1] == 0 and q[7] == 0 for q in qp)
    breakdown["assembly_ms"] = percentile([q[4] for q in qp])
    breakdown["gi_ms"] = percentile([q[5] for q in qp])
    ax = axes[1]
    labels = ["全身求解", "候選與停止核對", "狀態監看"]
    keys = ["solve_ms", "cand_ms", "mon_ms"]
    x = np.arange(3)
    ax.bar(x - .18, [breakdown[k]["p50"] for k in keys], width=.34,
           facecolor="white", edgecolor="black", hatch="///", label="中位數")
    ax.bar(x + .18, [breakdown[k]["p95"] for k in keys], width=.34,
           facecolor=".65", edgecolor="black", label="第 95 百分位數")
    ax.set_xticks(x, labels, fontproperties=FONT)
    ax.set_ylim(0, 34)
    cn_label(ax, y="主機時間（ms）", title="(b) 開門區段之內部計算分項")
    legend(ax, loc="upper right")
    export(fig, "control_computation_timing")
    seg = [x for x in simulator["camera"]["seg_ms"] if x is not None]
    alignment = saved["chain_align"]
    return {"representative_run": RUN.name, "response_count": len(rows), "stages": stats,
            "opening_breakdown_ms": breakdown, "wall_deadline_ms": 120,
            "normal_terminal_reply_count": sum(bool(r.get("ctrl_done")) and not r.get("commit") for r in rows),
            "simulation_clock_s": simulator["sim_s"], "host_wall_s": simulator["wall_s"],
            "real_time_factor": simulator["rtf"], "pacing_changes": simulator["pacing"]["changes"],
            "nominal_pacing_by_phase": simulator["pacing"]["sim_s_at"],
            "segmentation_ms": percentile(seg),
            "segmentation_max_is_first_sample": float(seg[0]) == max(seg),
            "proposal_release_audit": {k: alignment[k] for k in
                                        ("n_proposed", "n_effect_checked_ok", "n_missing", "n_differs", "delta_abs_max")},
            "data_scope": "all finite roundtrip replies from saved NPZ; startup without roundtrip excluded; one normal terminal reply included; QP breakdown only 189 opening cycles"}


FAILURE_CLASSES = {
    "FAIL not_ready": "回覆未就緒或外層逾時", "FAIL deadline": "回覆未就緒或外層逾時",
    "FAIL open_solver_fail": "開門求解或內部預算中止", "FAIL abort": "開門求解或內部預算中止",
    "FAIL lost_handle_during_pull": "抽屜抓持與退出條件未達成",
    "FAIL retreat_not_reached": "抽屜抓持與退出條件未達成",
    "FAIL handle_grasp_evidence_timeout": "抽屜抓持與退出條件未達成",
    "FAIL gripper_force_over": "接觸力保護中止"}


def failures():
    cohort = read(COHORT)
    failed = [r for r in cohort["batch"]["runs"] if r["verdict"] != "OK"]
    assert len(failed) == 14
    counts = {}
    details = []
    for run in failed:
        number = run["run"]
        base = ROOT / f"evaluation/runs/m8house_d2_unified_{number}"
        paths = [base / f for f in ("result.json", "d2_d2w.json", "d2_d2w.npz", "run.log")]
        SOURCES.extend(paths)
        result, control = read(paths[0]), read(paths[1])
        log = [json.loads(x) for x in np.load(paths[2])["log"]]
        category = FAILURE_CLASSES[run["verdict"]]
        counts[category] = counts.get(category, 0) + 1
        stop = result.get("stop_validation") or {}
        details.append({"run": number, "category": category, "verdict": run["verdict"],
                        "latch": result["latch"], "stop_validation": stop,
                        "tail_settled": control.get("settled_after_tail"),
                        "controller_stop": (control.get("wg") or {}).get("stop"),
                        "rejected_replies": [{k: row.get(k) for k in
                                              ("n", "ver", "mode", "phase", "compute_ms", "roundtrip_ms",
                                               "why", "commit", "stop")} for row in log if row.get("why")]})
    assert sorted(counts.values()) == [2, 3, 4, 5]
    engineering = [r for r in details if r["latch"] is not None]
    stops = [r for r in details if r["latch"] is None]
    assert len(engineering) == 2 and len(stops) == 12
    assert sum(bool(r["stop_validation"]["valid"]) for r in stops) == 10
    assert {r["run"] for r in stops if not r["stop_validation"]["valid"]} == {150, 179}
    return {"cohort_size": 43, "unsuccessful": 14, "classes": counts,
            "engineering_abort_without_post_abort_physics": 2,
            "tail_cases": 12, "stop_window_found_within_four_seconds": 10,
            "stop_window_not_found": 2, "details": details,
            "note": "classify each unsuccessful run once by stored terminal reason; not counts of all internal error events"}


def stopping_counterexamples():
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.3))
    fig.subplots_adjust(left=.14, right=.97, bottom=.11, top=.94, hspace=.43)
    info = []
    for ax, number, title in [(axes[0], 150, "(a) 抽屜拉開中止後"),
                              (axes[1], 179, "(b) 開門回覆逾時後")]:
        base = ROOT / f"evaluation/runs/m8house_d2_unified_{number}"
        path = base / "log.npz"
        SOURCES.append(path)
        archive = np.load(path)
        L = archive["L"]
        cols = {str(k): i for i, k in enumerate(archive["cols"])}
        saved = read(base / "result.json")["stop_validation"]
        t0 = saved["t_last_nonzero_input"]
        dt = np.diff(L[:, 0])
        joint_rate = abs(np.diff(L[:, 4:10], axis=0)).max(axis=1) / dt
        t = L[1:, 0]
        mask = t >= t0
        ax.plot(t[mask] - t0, joint_rate[mask] * 1000, color="black", lw=.9)
        ax.axhline(2, color=".4", ls="--", lw=.9, label="2 mrad/s 門檻")
        ax.axvline(4, color=".5", ls=":", lw=.8)
        cn_label(ax, x="自最後非零輸入之模擬時間（s）", y="最大逐軸關節速率（mrad/s）", title=title)
        ax.set_xlim(0, t[mask][-1] - t0)
        ax.set_ylim(0, max(joint_rate[mask]) * 1100)
        legend(ax, loc="upper right")
        # Independent search over the limiting arm condition alone. Since no
        # valid arm hold exists, the complete stopping condition cannot hold.
        arm_window_found = False
        for k in np.flatnonzero(mask & (t - t0 <= 4)):
            j = int(np.searchsorted(t, t[k] + .3))
            if j < len(t) and t[j] - t[k] >= .3 - 1e-6 and (joint_rate[k:j + 1] < .002).all():
                arm_window_found = True
                break
        assert not arm_window_found and not saved["valid"]
        info.append({"run": number, "last_nonzero_input_sim_s": t0,
                     "record_after_s": float(t[-1] - t0), "final_joint_rate_mrad_s": float(joint_rate[-1] * 1000),
                     "arm_only_window_found": arm_window_found, "saved_full_stop_valid": saved["valid"]})
    export(fig, "stopping_residual_counterexamples")
    return info


def main():
    data = {"timing": timing(), "failures": failures(),
            "stopping_counterexamples": stopping_counterexamples(),
            "new_simulator_runs": 0, "new_controller_runs": 0}
    data["sources"] = [{"path": str(p.relative_to(ROOT)),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(SOURCES))]
    (OUT / "performance_failure_limit_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"timing_stages": data["timing"]["stages"],
                      "opening_parts": data["timing"]["opening_breakdown_ms"],
                      "failure_counts": data["failures"]["classes"],
                      "stop_counterexamples": data["stopping_counterexamples"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
