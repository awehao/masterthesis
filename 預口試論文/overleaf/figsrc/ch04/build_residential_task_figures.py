"""Build Chapter 4.5 from existing residential logs and recorded RGB frames.

No simulator, inference or controller is launched. Cohort is fixed at 142--184.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from build_execution_model_figures import ROOT, OUT, FONT, cn_label, legend, export

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import yaml

REPRESENTATIVE = 173
RUN = ROOT / f"evaluation/runs/m8house_d2_unified_{REPRESENTATIVE}"
ANALYZER = ROOT / "第八次進度報告素材/產生程式/seed_batch_analyze.py"
REGISTRY = ROOT / "evaluation/experiments/20261118_m8_house_r2_scene.json"
SOURCES = [ANALYZER, REGISTRY,
           ROOT / "evaluation/experiments/20261119_m8_house_d2_unified173_registration.json",
           RUN / "log.npz", RUN / "result.json", RUN / "world.log",
           RUN / "sup.json", RUN / "frames_meta.json", RUN / "d2_d2w.json"]


def read(path):
    return json.loads(path.read_text())


def events(path):
    out = []
    for line in path.read_text().splitlines():
        if "{" not in line:
            continue
        try:
            out.append(json.loads(line[line.index("{"):]))
        except json.JSONDecodeError:
            pass
    return out


def first(ev, what, **conditions):
    return next(x for x in ev if x.get("what") == what and
                all(x.get(k) == v for k, v in conditions.items()))


def visual_targets(meta):
    choices = [("door_handle", 46.9, "wrist", "(a) 門把與門扇"),
               ("cylinder", 111.0, "base", "(b) 茶几圓柱"),
               ("drawer_handle", 164.9, "base", "(c) 抽屜面板與把手"),
               ("pen", 189.5, "wrist", "(d) 抽屜內筆狀物件")]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.8))
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.95, hspace=.22, wspace=.06)
    selected = []
    for ax, (cls, target_time, source, title) in zip(axes.flat, choices):
        candidates = [f for f in meta["frames"] if f.get("src") == source and
                      cls in (f.get("seg") or {}).get("cls", {})]
        f = min(candidates, key=lambda x: abs(x["t_obs"] - target_time))
        path = RUN / "frames" / f"{f['stem']}_rgb.png"
        SOURCES.append(path)
        with Image.open(path) as im:
            ax.imshow(np.asarray(im))
        ax.set_title(title, fontproperties=FONT, fontsize=12, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        selected.append({"class": cls, "source": source, "time_s": f["t_obs"],
                         "path": str(path.relative_to(ROOT)), "image_is_original_rgb": True})
    export(fig, "residential_visual_observations")
    return selected


def door_operation(L, col, ev, sup):
    t = L[:, col["t"]]
    take = first(ev, "d2_takeover")["t"]
    passed = first(ev, "pass_start")["t"]
    close_vision = first(sup["events"], "door_leaf_vision")
    close_end = first(ev, "pick_door_closed")["t"]
    grasp = first(ev, "door_bind2_committed")["t"]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2))
    fig.subplots_adjust(left=.14, right=.97, bottom=.11, top=.94, hspace=.40)
    for ax, start, stop, title in [
            (axes[0], take, passed, "(a) 接近與開門"),
            (axes[1], close_vision["t"], close_end, "(b) 回程再觀測與關門")]:
        mask = (t >= start - 1e-5) & (t <= stop + .05)
        ax.plot(t[mask] - start, L[mask, col["door_deg"]], color="black", lw=1.2)
        cn_label(ax, x="自該階段起點經過之模擬時間（s）", y="實測門角（度）", title=title)
        ax.set_xlim(0, stop - start + .5)
        ax.set_ylim(-3, 100)
    for ax, start, tm, label in [
            (axes[0], take, first(ev, "bind_committed")["t"], "抓持建立"),
            (axes[1], close_vision["t"], grasp, "再次抓持")]:
        ax.axvline(tm - start, color=".45", lw=.8, ls="--")
        ax.text(tm - start - .5, 94, label, fontproperties=FONT, fontsize=10,
                ha="right", va="top")
    axes[1].plot([0], [close_vision["phi_deg"]], marker="o", mfc="white", mec="black", ms=5,
                 color="black", linestyle="none",
                 label="視覺門角估計")
    legend(axes[1], loc="lower left")
    export(fig, "residential_door_open_close")
    k = int(np.argmin(abs(t - close_vision["t"])))
    return {"opening_window_s": [take, passed], "closing_window_s": [close_vision["t"], close_end],
            "vision_angle_deg": close_vision["phi_deg"], "true_angle_at_vision_deg": L[k, col["door_deg"]],
            "vision_angle_error_deg": abs(close_vision["phi_deg"] - L[k, col["door_deg"]])}


def cylinder_operation(L, col, result, ev):
    start = first(ev, "pick_start")["t"]
    closed = first(ev, "pick_phase", phase="CLOSE")["t"]
    attached = first(ev, "pick_phase", phase="LIFT")["t"]
    binned = first(ev, "pick_binned")["t"]
    t = L[:, col["t"]]
    # 100 ms displacement average, centered at each physical sample.
    v = np.full(len(t), np.nan)
    v[5:-5] = np.linalg.norm(L[10:, 1:3] - L[:-10, 1:3], axis=1) / (t[10:] - t[:-10])
    mask = (t >= start - 1e-5) & (t <= binned + 1e-5)
    data = np.array(result["cyl"]["log"], float)
    ci = {k: i for i, k in enumerate(result["cyl"]["log_cols"])}
    cmask = (data[:, ci["t"]] >= start) & (data[:, ci["t"]] <= binned)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.1), sharex=True)
    fig.subplots_adjust(left=.14, right=.97, bottom=.11, top=.94, hspace=.36)
    axes[0].plot(t[mask] - start, v[mask], color="black", lw=1.1)
    cn_label(axes[0], y="底盤平移速率（m/s）", title="(a) 移動中接近與閉爪")
    axes[0].set_ylim(bottom=0)
    axes[1].plot(data[cmask, ci["t"]] - start, data[cmask, ci["z"]],
                 color="black", lw=1.1, label="圓柱中心")
    axes[1].plot(data[cmask, ci["t"]] - start, data[cmask, ci["grip_z"]],
                 color=".35", lw=1.1, ls="--", label="TCP")
    cn_label(axes[1], x="自圓柱接近開始之模擬時間（s）", y="世界座標高度（m）",
             title="(b) 抬升與車尾放置")
    legend(axes[1], loc="lower left", ncol=2)
    for ax in axes:
        ax.axvspan(closed - start, attached - start, color=".87", zorder=0)
        ax.set_xlim(0, binned - start)
    axes[0].annotate("閉爪至確認", xy=((closed + attached) / 2 - start, .04),
                     xytext=(4.0, .11), fontproperties=FONT, fontsize=10,
                     arrowprops={"arrowstyle": "->", "lw": .7, "color": "black"})
    export(fig, "residential_cylinder_mobile_pick")
    k = np.flatnonzero((t >= closed - 1e-5) & (t <= attached + 1e-5))
    path = float(np.linalg.norm(np.diff(L[k, 1:3], axis=0), axis=1).sum())
    return {"start_s": start, "close_s": closed, "attach_s": attached, "binned_s": binned,
            "close_to_attach_duration_s": float(t[k[-1]] - t[k[0]]),
            "base_path_close_to_attach_m": path,
            "base_mean_speed_close_to_attach_mps": path / (t[k[-1]] - t[k[0]]),
            "close_event": first(ev, "pick_phase", phase="CLOSE")}


def drawer_operation(L, col, result, ev):
    start = first(ev, "pick_dr_start")["t"]
    end = first(ev, "pick_dr_closed")["t"]
    data = np.array(result["drawer"]["log"], float)
    dc = {k: i for i, k in enumerate(result["drawer"]["log_cols"])}
    mask = (data[:, dc["t"]] >= start) & (data[:, dc["t"]] <= end)
    open_ev = first(ev, "pick_dr_pulled")
    # Physical readback events distinguish actual opening from controller estimate.
    unbound = [x for x in result["events"] if x["what"] == "drawer_unbind_fixed_joint"]
    obs = first(ev, "pick_dr_observing")["t"]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2))
    fig.subplots_adjust(left=.14, right=.97, bottom=.11, top=.94, hspace=.40)
    ax = axes[0]
    ax.plot(data[mask, dc["t"]] - start, 1000 * data[mask, dc["opening_truth"]],
            color="black", lw=1.2)
    ax.axhline(200, color=".55", ls="--", lw=.75)
    for tm, text in [(open_ev["t"], "拉開完成"), (obs, "觀察筆"),
                     ([x for x in ev if x.get("what") == "pick_binned"][1]["t"], "筆入盒")]:
        ax.axvline(tm - start, color=".6", ls=":", lw=.7)
        ax.text(tm - start + .7, 225, text, fontproperties=FONT, fontsize=10,
                ha="left", va="top")
    cn_label(ax, x="自抽屜接近開始之模擬時間（s）", y="抽屜開度（mm）", title="(a) 開抽屜、取筆與推回")
    ax.set_ylim(-10, 250)
    ax.set_xlim(0, end - start)
    ax = axes[1]
    # Project measured base translation along the visually estimated pull axis.
    app = next(x for x in ev if x.get("what") == "pick_app" and x.get("app", {}).get("kind") == "drawer")
    axis = np.asarray(app["app"]["pull"], float)
    pull_start = first(ev, "pick_phase", phase="PULL")["t"]
    t = L[:, col["t"]]
    k = np.flatnonzero((t >= pull_start - 1e-5) & (t <= open_ev["t"] + 1e-5))
    displacement = (L[k, 1:3] - L[k[0], 1:3]) @ axis * 1000
    ax.plot(t[k] - pull_start, displacement, color="black", ls="--", lw=1.2, label="底盤軸向平移")
    q_open = np.interp(t[k], data[:, dc["t"]], data[:, dc["opening_truth"]])
    ax.plot(t[k] - pull_start, (q_open - q_open[0]) * 1000, color="black", lw=1.2,
            label="抽屜開度變化")
    cn_label(ax, x="自拉開開始之模擬時間（s）", y="軸向位移（mm）", title="(b) 拉開期間之底盤移動")
    ax.set_xlim(0, open_ev["t"] - pull_start)
    ax.set_ylim(bottom=0)
    legend(ax, loc="upper left")
    export(fig, "residential_drawer_pick_close")
    return {"start_s": start, "end_s": end, "opening_readbacks": unbound,
            "final_opening_m": float(data[-1, dc["opening_truth"]]),
            "pull_base_axial_m": float(displacement[-1] / 1000)}


def mission_overview(L, col, result, ev):
    reg = read(REGISTRY)
    doorpath = ROOT / reg["door"]["asset"]
    SOURCES.append(doorpath)
    door = yaml.safe_load(doorpath.read_text())
    t = L[:, col["t"]]
    start = first(ev, "d2_takeover")["t"]
    ends = [first(ev, "pass_start")["t"], first(ev, "pick_binned")["t"],
            [x for x in ev if x.get("what") == "pick_binned"][1]["t"],
            first(ev, "pick_dr_closed")["t"], first(ev, "pick_door_closed")["t"],
            first(ev, "world_end")["t"]]
    labels = ["開門", "穿門至圓柱收納", "抽屜取筆與收納", "關抽屜", "回程通行與關門", "返回起點"]
    fig = plt.figure(figsize=(6.4, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], left=.23, right=.97,
                          bottom=.10, top=.95, hspace=.40)
    ax = fig.add_subplot(grid[0])

    def footprint(item, color):
        c, s = item["c"], item["s"]
        ax.add_patch(Rectangle((c[1] - s[1] / 2, -c[0] - s[0] / 2), s[1], s[0],
                               facecolor=color, edgecolor=".45", lw=.4))
    for x in reg["walls"]:
        footprint(x, ".3")
    for x in reg["furniture"]:
        footprint(x, ".88")
    lo, hi = door["wall"]["extent_x"]
    gap = door["frame_clear_width"]
    th = door["wall"]["thickness"]
    for a, b in [(lo, -door["wall"].get("hinge_side_recess", 0.)), (gap, hi)]:
        footprint({"c": [(a + b) / 2, th / 2, 0], "s": [b - a, th, 0]}, ".3")
    # Actual shifted shallow drawer, not the older registered full-depth cabinet.
    dp = result["drawer"]["pose_true"]
    ax.add_patch(Rectangle((dp[1] - .2875, -dp[0] - .15), .575, .30, fc=".75", ec="black", lw=.5))
    end = first(ev, "world_end")["t"]
    m = (t >= start) & (t <= end + 1e-5)
    route = L[m][::10]
    ax.plot(route[:, 2], -route[:, 1], color="black", lw=.8)
    origin = L[np.argmin(abs(t - start)), 1:3]
    ax.plot(origin[1], -origin[0], marker="o", mec="black", mfc="white", ms=5)
    for label, xy in [("起終點", (-4.3, -2.8)), ("通行門", (-1.0, 1.3)),
                      ("茶几", (4.5, -3.4)), ("抽屜", (10.0, 3.4))]:
        ax.text(*xy, label, fontproperties=FONT, fontsize=10, ha="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})
    ax.set_aspect("equal")
    ax.set_xlim(-7.4, 13.)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlabel("世界座標 y（m）", fontproperties=FONT, fontsize=11)
    ax.set_ylabel("負世界座標 x（m）", fontproperties=FONT, fontsize=11)
    ax.set_title("(a) 同一次任務之底盤軌跡", fontproperties=FONT, fontsize=12, pad=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax = fig.add_subplot(grid[1])
    starts = [start] + ends[:-1]
    durations = np.array(ends) - starts
    for i, (a, b) in enumerate(zip(starts, ends)):
        ax.barh(i, b - a, left=a - start, height=.62, facecolor=["white", ".7"][i % 2],
                edgecolor="black", hatch=["///", None][i % 2], lw=.8)
        ax.text(b - start + 3, i, f"{b - a:.1f}", va="center", fontsize=10)
    ax.set_yticks(np.arange(6), labels, fontproperties=FONT)
    ax.invert_yaxis()
    ax.set_xlim(0, 362)
    cn_label(ax, x="自控制接手之模擬時間（s）", title="(b) 任務階段與區間長度")
    ax.grid(False)
    ax.grid(axis="x", color=".85", lw=.5)
    export(fig, "residential_mission_trajectory_timeline")
    return {"takeover_sim_s": start, "world_end_sim_s": end,
            "duration_s": end - start, "stage_labels": labels,
            "stage_starts_sim_s": starts, "stage_ends_sim_s": ends,
            "stage_durations_s": durations.tolist(),
            "end_distance_to_start_m": float(np.linalg.norm(L[-1, 1:3] - [1.5, -4.5]))}


def batch_comparison():
    spec = importlib.util.spec_from_file_location("residential_batch", ANALYZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = []
    for number in range(142, 185):
        run = module.analyze(number)
        base = ROOT / f"evaluation/runs/m8house_d2_unified_{number}"
        reg = ROOT / f"evaluation/experiments/20261119_m8_house_d2_unified{number}_registration.json"
        SOURCES.extend([base / "world.log", base / "sup.log", base / "log.npz", base / "result.json", reg])
        result = read(base / "result.json")
        assert not result["partial"]
        run["has_walkers"] = bool(result.get("walkers"))
        run["end_in_bin"] = [x["in_bin"] for x in result["bin"]["end"]]
        if run["verdict"] == "OK":
            assert run["end_in_bin"] == [True, True]
        runs.append(run)
    nowalk = runs[:27]
    walk = runs[27:]
    assert all(not x.get("has_walkers") for x in nowalk)
    assert all(x.get("has_walkers") for x in walk)
    seeds = list(range(2, 10))
    paired = []
    for seed in seeds:
        row = {"seed": seed, "methods": {}}
        for mode in "MP":
            group = [r for r in nowalk if r["mode"] == mode and r["seed"] == seed and r["verdict"] == "OK"]
            assert group
            row["methods"][mode] = {"runs": [r["run"] for r in group], **{
                key: float(np.mean([r[key] for r in group])) for key in
                ("mission_s", "still_s", "still_arm_s", "moving_arm_s", "jerk_p95")}}
        paired.append(row)
    means = {mode: {key: float(np.mean([r["methods"][mode][key] for r in paired]))
                    for key in ("mission_s", "still_s", "still_arm_s", "moving_arm_s", "jerk_p95")}
             for mode in "MP"}
    assert abs(means["M"]["mission_s"] - 319.25) < 1e-9
    assert abs(means["P"]["mission_s"] - 337.53125) < 1e-9
    counts = {condition: {mode: {"attempted": sum(r["mode"] == mode for r in rr),
                                "completed": sum(r["mode"] == mode and r["verdict"] == "OK" for r in rr)}
                          for mode in "MP"} for condition, rr in [("no_walkers", nowalk), ("with_walkers", walk)]}
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2))
    fig.subplots_adjust(left=.14, right=.97, bottom=.11, top=.94, hspace=.44)
    ax = axes[0]
    x = np.arange(8)
    for shift, mode, fc, hatch, label in [(-.18, "M", "white", "///", "移動中操作"),
                                          (.18, "P", ".65", None, "停駐後操作")]:
        ax.bar(x + shift, [r["methods"][mode]["mission_s"] for r in paired], .34,
               color=fc, edgecolor="black", hatch=hatch, lw=.65, label=label)
    ax.set_xticks(x, seeds)
    ax.set_ylim(0, 405)
    cn_label(ax, x="圓柱位置種子", y="任務時間（s）", title="(a) 無行人完成案例之條件時間比較")
    legend(ax, loc="upper left", ncol=2)
    ax = axes[1]
    keys = ["still_s", "still_arm_s", "moving_arm_s"]
    x = np.arange(3)
    for shift, mode, fc, hatch, label in [(-.18, "M", "white", "///", "移動中操作"),
                                          (.18, "P", ".65", None, "停駐後操作")]:
        ax.bar(x + shift, [means[mode][k] for k in keys], .34,
               color=fc, edgecolor="black", hatch=hatch, lw=.65, label=label)
    ax.set_xticks(x, ["底盤低速", "底盤低速且動臂", "底盤移動且動臂"], fontproperties=FONT)
    ax.set_ylim(0, 100)
    cn_label(ax, y="種子等權平均時間（s）", title="(b) 底盤與手臂運動區間")
    legend(ax, loc="upper right", ncol=2)
    export(fig, "residential_strategy_seed_comparison")
    return {"cohort_start_run": 142, "cohort_end_run": 184,
            "counts": counts, "runs": runs, "paired_seeds": paired, "no_walkers_means": means,
            "reduction_pct": (means["P"]["mission_s"] - means["M"]["mission_s"]) / means["P"]["mission_s"] * 100,
            "aggregation": "completed cases only for time; within seed/mode average repetitions first, then equal seed weights",
            "version_scope": "development integration records include revisions; counts are descriptive, not frozen-controller success rates",
            "low_speed_definition": "from 50 ms position differences: speed < 0.01 m/s and abs yaw rate < 0.02 rad/s",
            "arm_moving_definition": "norm of six joint velocity vector > 0.05 rad/s",
            "simultaneous_moving_definition": "base speed > 0.03 m/s and arm_moving",
            "completion": "return-door-closed event, two binned events, final base within 0.6 m of start; completed cases also checked for both end-in-bin flags"}


def main():
    result = read(RUN / "result.json")
    meta = read(RUN / "frames_meta.json")
    sup = read(RUN / "sup.json")
    # Console JSON is truncated to 500 characters by the writer. The saved
    # structured events retain full objects, including approach definitions.
    ev = read(RUN / "d2_d2w.json")["events"]
    archive = np.load(RUN / "log.npz")
    L = archive["L"]
    col = {str(k): i for i, k in enumerate(archive["cols"])}
    data = {"representative_run": REPRESENTATIVE, "representative_selection": "same integrated with-walker case used in existing progress report; illustrative, not cohort estimate",
            "visual_frames": visual_targets(meta), "door": door_operation(L, col, ev, sup),
            "cylinder": cylinder_operation(L, col, result, ev),
            "drawer": drawer_operation(L, col, result, ev),
            "mission": mission_overview(L, col, result, ev), "batch": batch_comparison(),
            "end": {"base_xytheta": L[-1, 1:4].tolist(), "door_deg": float(L[-1, col["door_deg"]]),
                    "objects": result["bin"]["end"]},
            "new_simulator_runs": 0, "new_network_inference": 0}
    a, b = data["mission"]["takeover_sim_s"], first(ev, "app_start")["t"]
    errors = [x["eval_only"]["err_pos_mm"] for x in meta["frames"] if a <= x["t_obs"] <= b and
              x.get("valid") and x.get("eval_only", {}).get("err_pos_mm") is not None]
    data["door"]["handle_observation_error_mm"] = {"window_sim_s": [a, b], "n": len(errors),
                                                     "median": float(np.median(errors)), "max": max(errors)}
    data["sources"] = [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(set(SOURCES))]
    (OUT / "residential_task_validation_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figures": 6, "counts": data["batch"]["counts"],
                      "mission": data["mission"], "door_error": data["door"]["handle_observation_error_mm"],
                      "cylinder_base_speed": data["cylinder"]["base_mean_speed_close_to_attach_mps"],
                      "end": data["end"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
