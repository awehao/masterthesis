"""Chapter 4.4: figures and provenance from archived drawer experiments only."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

from build_execution_model_figures import OUT, ROOT, FONT, cn_label, legend, export
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

sys.path.insert(0, str(ROOT / "src/ammr_wholebody_mpc"))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics

PAIR_IDS = ["p1", "p2", "p3r"]
HORIZON = ROOT / "evaluation/results/horizon_ablation/c0b1_summary.json"
STRATEGY = ROOT / "evaluation/results/motm_speed/park_hold_formal_results.json"
KINEMATICS = ROOT / "evaluation/models/omni_bot_wholebody_expanded.urdf"
SOURCES = [HORIZON, STRATEGY, KINEMATICS,
           ROOT / "src/ammr_wholebody_mpc/ammr_wholebody_mpc/wholebody_kinematics.py",
           ROOT / "src/my_omnibot_description/config/drawer_unit_bar26.yaml",
           ROOT / "evaluation/results/horizon_ablation/H1_H5_resolved_config.yaml",
           ROOT / "evaluation/results/horizon_ablation/c0b1_results.yaml",
           ROOT / "evaluation/results/motm_speed/park_hold_formal_plan.yaml",
           ROOT / "evaluation/results/motm_speed/park_hold_v21_dev_config.yaml",
           ROOT / "evaluation/results/motm_speed/park_hold_formal_summary.yaml",
           ROOT / "evaluation/archive/horizon_metrics.py",
           ROOT / "evaluation/archive/motm_metrics.py"]


def read(path):
    return json.loads(path.read_text())


def coordination():
    run = ROOT / "evaluation/runs/c0b1_p1_H5"
    paths = [run / x for x in ("room_run.json", "task.json", "align_solver.json",
                              "analysis/horizon_metrics.json")]
    SOURCES.extend(paths)
    sim, task, solver, metrics = map(read, paths)
    K = WholeBodyKinematics.from_urdf_file(str(KINEMATICS))
    col = {k: i for i, k in enumerate(sim["steps_cols"])}
    times = np.array([s[col["sim_t"]] for s in sim["steps"]])
    starts = {e["phase"]: e["sim_t"] for e in task["events"] if "phase" in e}
    begin, end = [metrics["window"][k] for k in ("from_sim_t", "to_sim_t")]
    k0 = int(np.argmin(abs(times - starts["OPEN"])))
    b0 = sim["steps"][k0][col["base_xyth"]]
    q0 = sim["steps"][k0][col["q_arm_meas"]]
    p0 = K.fk(np.r_[b0, q0], "link_tcp")[:3, 3]
    axis = np.array([0., -1., 0.])
    records = []
    # Whole contact operation, from OPEN through RELEASE_WAIT, fixed in advance.
    for j in np.flatnonzero((times >= times[k0]) & (times <= starts["RELEASE_WAIT"])):
        row = sim["steps"][j]
        b, q = row[col["base_xyth"]], row[col["q_arm_meas"]]
        pe = K.fk(np.r_[b, q], "link_tcp")[:3, 3]
        pb = K.fk(np.r_[b, q0], "link_tcp")[:3, 3]
        total, base, arm = [float(axis @ v * 1000) for v in
                            (pe - p0, pb - p0, pe - pb)]
        assert abs(total - base - arm) < 1e-9
        records.append([float(times[j] - begin), total, base, arm,
                        float(row[col["opening_m"]] * 1000)])
    a = np.array(records)
    logs = [x for x in solver["log"] if begin - 1e-5 <= x["sim_t"] <= end
            and "solve_in" in x and x.get("err_p") is not None]
    lt = np.array([x["sim_t"] - begin for x in logs])
    error = np.array([x["err_p"] * 1000 for x in logs])
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.7), sharex=True)
    fig.subplots_adjust(left=.14, right=.97, top=.94, bottom=.10, hspace=.32)
    ax = axes[0]
    ax.plot(a[:, 0], a[:, 4], color=".65", lw=3, label="抽屜開度")
    ax.plot(a[:, 0], a[:, 1], color="black", lw=1.25, label="TCP 總位移")
    ax.plot(a[:, 0], a[:, 2], color="black", lw=1.25, ls="--", label="底盤幾何分量")
    ax.plot(a[:, 0], a[:, 3], color="black", lw=1.25, ls=":", label="手臂幾何分量")
    cn_label(ax, y="沿開啟方向之位移（mm）", title="(a) 抽屜操作之位移分工")
    legend(ax, loc="upper left", ncol=2, fontsize=10)
    ax.set_ylim(-20, 280)
    ax = axes[1]
    ax.plot(lt, error, color="black", lw=1.0)
    cn_label(ax, x="自全身控制開始之模擬時間（s）", y="位置誤差（mm）",
             title="(b) 末端位置追蹤誤差")
    ax.set_ylim(0, 14)
    for ax in axes:
        for phase in ("OPEN", "CLOSE"):
            ax.axvline(starts[phase] - begin, color=".45", ls="-.", lw=.8)
        ax.set_xlim(0, end - begin)
        ax.set_xticks(np.arange(0, 31, 5))
    axes[1].text(1.0, 12.4, "接近與夾持", fontproperties=FONT, fontsize=11)
    axes[1].text(11.3, 12.4, "拉開與保持", fontproperties=FONT, fontsize=11)
    axes[1].text(20.1, 12.4, "推回與退出", fontproperties=FONT, fontsize=11)
    export(fig, "wholebody_drawer_coordination")
    k_hold = int(np.argmin(abs(a[:, 0] - (starts["OPEN_HOLD"] - begin))))
    return {"run": run.name, "selection": "first registered pair, H5; no best-case selection",
            "window_sim_s": [begin, end], "opening_direction_world": axis.tolist(),
            "decomposition": "base changes first at q0, then arm changes at current base; exact telescoping FK difference",
            "open_hold_start": {"time_from_control_start_s": a[k_hold, 0],
                                "tcp_mm": a[k_hold, 1], "base_mm": a[k_hold, 2],
                                "arm_mm": a[k_hold, 3], "opening_mm": a[k_hold, 4]},
            "columns": ["time_s", "tcp_mm", "base_mm", "arm_mm", "opening_mm"],
            "trajectory": records,
            "tracking_time_s": lt.tolist(), "tracking_position_mm": error.tolist()}


def horizon_comparison():
    s = read(HORIZON)
    keys = [("err_p_p95_m", 1000, "(a) 位置追蹤誤差", "第 95 百分位數（mm）"),
            ("du_arm_p95", 1, "(b) 手臂命令增量", "第 95 百分位數（rad/s）"),
            ("window_s", 1, "(c) 全身控制區間", "模擬時間（s）"),
            ("core_ms_p50", 1, "(d) 核心計算耗時", "中位數（ms）")]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4))
    fig.subplots_adjust(left=.12, right=.98, top=.91, bottom=.12, wspace=.55, hspace=.55)
    for ax, (key, scale, title, unit) in zip(axes.flat, keys):
        for i, pair in enumerate(PAIR_IDS):
            values = [s["runs"][f"c0b1_{pair}_{method}"]["m"][key] * scale
                      for method in ("H1", "H5")]
            ax.plot([0, 1], values, marker=["o", "s", "^"][i], color="black",
                    ls=["-", "--", ":"][i], ms=5, lw=1,
                    markerfacecolor=["white", ".5", "black"][i], label=f"案例{['一', '二', '三'][i]}")
        cn_label(ax, y=unit, title=title)
        ax.set_xticks([0, 1], ["單步", "五步"], fontproperties=FONT)
        ax.set_xlim(-.3, 1.3)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               prop=FontProperties(fname=str(ROOT / "預口試論文/overleaf/fonts/kaiu.ttf"), size=10),
               frameon=False, bbox_to_anchor=(.55, 1.0))
    export(fig, "horizon_tracking_comparison")
    runs = {}
    for pair in PAIR_IDS:
        for method in ("H1", "H5"):
            rid = f"c0b1_{pair}_{method}"
            source = ROOT / f"evaluation/runs/{rid}/analysis/horizon_metrics.json"
            SOURCES.append(source)
            hm = read(source)
            assert hm["N"] == (1 if method == "H1" else 5)
            run = s["runs"][rid]
            assert run["flow_final_phase"] == "DONE" and run["physical"] == "PASS"
            assert abs(run["m"]["err_p_p95_m"] - hm["by_phase"]["WINDOW"]["tracking"]["err_p_m"]["p95"]) < 1e-12
            runs[rid] = run
    return {"pairs": PAIR_IDS, "runs": runs,
            "command_difference_definition": "max over six joints of absolute difference between adjacent solver request_body outputs; no division by dt; rad/s",
            "reversal_definition": "within phase, both adjacent commands abs > 0.05 rad/s and opposite signs; sum over six joints divided by window duration",
            "window_definition": "first solver cycle through solver stop after retreat; excludes entry navigation, unfolding, stowing and return",
            "same_target_over_horizon": True, "terminal_task_multiplier": 5,
            "scope": "three paired archived drawer experiments; solver.N is only configuration difference, but changes cost accumulation and terminal placement"}


def strategy_comparison():
    s = read(STRATEGY)
    runs = {x["rid"]: x for x in s["runs"]}
    pair_names = list(s["pairs"])
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.6))
    fig.subplots_adjust(left=.22, right=.97, top=.95, bottom=.12, hspace=.55)
    ax = axes[0]
    x = np.arange(3)
    moving = [runs[s["pairs"][p]["M"]]["T_complete_s"] for p in pair_names]
    parked = [runs[s["pairs"][p]["H"]]["T_complete_s"] for p in pair_names]
    ax.bar(x - .18, moving, width=.34, color="white", edgecolor="black", hatch="///", label="移動中操作")
    ax.bar(x + .18, parked, width=.34, color=".65", edgecolor="black", label="停駐後操作")
    for xx, v in zip(x - .18, moving):
        ax.text(xx, v + 1.1, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    for xx, v in zip(x + .18, parked):
        ax.text(xx, v + 1.1, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    cn_label(ax, y="完整任務時間（s）", title="(a) 共同起終事件之配對比較")
    ax.set_xticks(x, ["案例一", "案例二", "案例三"], fontproperties=FONT)
    ax.set_ylim(0, 115)
    legend(ax, ncol=2, loc="upper left")
    ax = axes[1]
    segment_keys = ["transfer_to_wholebody", "grasp_attached", "open_hold_done",
                    "close_hold_done", "handback_to_nav", "terminal_complete"]
    labels = ["進入操作區", "接手至夾持", "拉開及保持", "推回及保持", "釋放與收臂", "返回起點"]
    differences = []
    for name in pair_names:
        pair = s["pairs"][name]
        m, h = runs[pair["M"]], runs[pair["H"]]
        assert pair["usable"] and m["task_ok"] and h["task_ok"] and h["mode_ok"]
        mv = np.diff([0] + [m["segments"][k] for k in segment_keys])
        hv = np.diff([0] + [h["segments"][k] for k in segment_keys])
        diff = hv - mv  # positive means time saved by moving strategy
        assert abs(diff.sum() - (h["T_complete_s"] - m["T_complete_s"])) < 1e-8
        differences.append(diff)
    means = np.mean(differences, axis=0)
    ax.barh(np.arange(6), means, color=["white" if x >= 0 else ".6" for x in means],
            edgecolor="black", height=.65)
    for i, value in enumerate(means):
        ax.text(value + (.13 if value >= 0 else -.13), i, f"{value:.2f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=10)
    ax.set_yticks(np.arange(6), labels, fontproperties=FONT)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=.75)
    ax.set_xlim(-3.5, 13.1)
    cn_label(ax, x="停駐時間減去移動時間之配對平均（s）", title="(b) 各階段對時間差之貢獻")
    ax.grid(False)
    ax.grid(axis="x", color=".85", lw=.55)
    export(fig, "moving_parked_drawer_comparison")
    return {"runs": s["runs"], "pairs": s["pairs"],
            "segment_keys": segment_keys, "segment_labels": labels,
            "paired_saved_time_by_stage_s": np.array(differences).tolist(),
            "mean_saved_time_by_stage_s": means.tolist(),
            "mean_full_time_s": {"moving": float(np.mean(moving)), "parked": float(np.mean(parked))},
            "time_basis": "simulation clock; first navigation-plan publication to independent terminal completion",
            "scope": "whole-strategy comparison, not isolated base motion; alternating order M-H, H-M, M-H"}


def main():
    data = {"coordination": coordination(), "horizon": horizon_comparison(),
            "strategy": strategy_comparison()}
    unique = sorted(set(SOURCES))
    data["sources"] = [{"path": str(p.relative_to(ROOT)),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in unique]
    data["new_simulation_runs"] = 0
    (OUT / "wholebody_operation_comparison_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figures": ["wholebody_drawer_coordination.png", "horizon_tracking_comparison.png",
                                  "moving_parked_drawer_comparison.png"],
                      "mean_strategy_times_s": data["strategy"]["mean_full_time_s"],
                      "mean_stage_savings_s": data["strategy"]["mean_saved_time_by_stage_s"],
                      "coordination_endpoint": data["coordination"]["open_hold_start"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
