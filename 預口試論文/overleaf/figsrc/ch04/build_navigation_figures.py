"""Chapter 4.3: read archived navigation CSVs and scene inputs; no simulation."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis_ch4_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FONT_PATH = ROOT / "預口試論文/overleaf/fonts/kaiu.ttf"
FONT = FontProperties(fname=str(FONT_PATH))
FONT10 = FontProperties(fname=str(FONT_PATH), size=10)
INPUT = ROOT / "evaluation/results/manifests/gmpc_cbf__scan_seed1/inputs"
METHODS = ["gmpc100", "mppi100", "rpp100"]
ABLATION = ["ablA", "ablB", "ablC", "gmpc100"]
LABELS = {"gmpc100": "本研究導航", "mppi100": "MPPI", "rpp100": "RPP"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                     "mathtext.fontset": "stix", "svg.fonttype": "path",
                     "savefig.facecolor": "white", "axes.unicode_minus": False})


def export(fig, name):
    fig.canvas.draw()
    outside_ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick.get_loc() < lo or tick.get_loc() > hi:
                    outside_ticks.update((tick.label1, tick.label2))
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text() or item in outside_ticks:
            continue
        bounds = item.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > width + 1 or bounds.y1 > height + 1:
            raise ValueError(f"Clipped text in {name}: {item.get_text()}")
    fig.savefig(OUT / f"{name}.png", dpi=400)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontproperties=FONT, fontsize=12)
    ax.set_ylabel(ylabel, fontproperties=FONT, fontsize=12)
    ax.set_title(title, fontproperties=FONT, fontsize=13, pad=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.85", linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)


def load_rows(key):
    path = ROOT / "evaluation/results" / key / "batch.csv"
    rows = {}
    for row in csv.DictReader(path.open()):
        seed = int(row["run"].split("seed")[-1])
        if seed in rows:
            raise ValueError(f"Duplicate route: {key}/{seed}")
        rows[seed] = row
    return rows


def finite_clearance(row):
    return math.isfinite(float(row["min_clearance_m"]))


def summarize(rows):
    arrived = [row for row in rows if row["success"] == "True"]
    finite = [row for row in rows if finite_clearance(row)]
    return {"n_routes": len(rows), "n_arrived": len(arrived), "n_clearance": len(finite),
            "n_overlap_routes": sum(float(row["min_clearance_m"]) < 0 for row in finite),
            "collision_event_sum_diagnostic_only": sum(int(row["collision_count"]) for row in finite),
            "min_clearance_median_m": statistics.median(float(row["min_clearance_m"]) for row in finite),
            "min_clearance_worst_m": min(float(row["min_clearance_m"]) for row in finite),
            "arrival_time_median_s_arrived_only": statistics.median(float(row["arrival_time_s"]) for row in arrived),
            "path_length_median_m": statistics.median(float(row["path_length_m"]) for row in rows)}


def extract_metrics():
    data = {key: load_rows(key) for key in set(METHODS + ABLATION)}
    common_arrival = sorted(set.intersection(*(set(data[key]) for key in METHODS)))
    common_clearance = [i for i in common_arrival if all(finite_clearance(data[key][i]) for key in METHODS)]
    common_ablation = sorted(set.intersection(*(set(data[key]) for key in ABLATION)))
    common_ablation = [i for i in common_ablation if all(finite_clearance(data[key][i]) for key in ABLATION)]
    assert len(common_arrival) == 83
    assert len(common_clearance) == 82
    assert len(common_ablation) == 28
    arrival = {key: summarize([data[key][i] for i in common_arrival]) for key in METHODS}
    clearance = {key: summarize([data[key][i] for i in common_clearance]) for key in METHODS}
    ablation = {key: summarize([data[key][i] for i in common_ablation]) for key in ABLATION}
    assert [arrival[key]["n_arrived"] for key in METHODS] == [83, 64, 60]
    assert [clearance[key]["n_overlap_routes"] for key in METHODS] == [1, 6, 30]
    assert [ablation[key]["n_overlap_routes"] for key in ABLATION] == [22, 3, 2, 0]
    assert all(ablation[key]["n_arrived"] == 28 for key in ABLATION)
    excluded = [{"route": i, "missing_methods": [key for key in METHODS if not finite_clearance(data[key][i])],
                 "kept_in_arrival_analysis": True} for i in common_arrival if i not in common_clearance]
    return data, common_arrival, common_clearance, common_ablation, arrival, clearance, ablation, excluded


def scene_plot():
    # The archived XML comments contain double hyphens. Strip comments for
    # parsing, leaving all model/collision geometry unchanged on disk.
    xml = re.sub(r"<!--.*?-->", "", (INPUT / "bigarena.sdf").read_text(), flags=re.S)
    world = ET.fromstring(xml).find("world")
    trajectories = yaml.safe_load((INPUT / "dynamic_trajectories_bigarena_traffic.yaml").read_text())["dynamic_obstacles"]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    fig.subplots_adjust(left=.12, right=.98, bottom=.18, top=.98)
    ax.set_aspect("equal")
    for model in world.findall("model"):
        name = model.get("name", "")
        if name == "ground_plane":
            continue
        pose = [float(x) for x in model.findtext("pose", "0 0 0 0 0 0").split()]
        if name.startswith("wall"):
            face, hatch, lw = "0.22", None, .5
        elif name.startswith("known_obs"):
            face, hatch, lw = "0.78", None, .6
        elif name.startswith("unknown_obs"):
            face, hatch, lw = "white", "///", .6
        elif name.startswith("dyn_obs"):
            face, hatch, lw = "white", None, 1.0
        else:
            continue
        for col in model.findall("./link/collision"):
            offset = [float(x) for x in col.findtext("pose", "0 0 0 0 0 0").split()]
            x, y = pose[0] + offset[0], pose[1] + offset[1]
            size = col.findtext("./geometry/box/size")
            radius = col.findtext("./geometry/cylinder/radius")
            if size:
                sx, sy, _ = map(float, size.split())
                ax.add_patch(Rectangle((x-sx/2, y-sy/2), sx, sy,
                                       fc=face, ec="0.12", hatch=hatch, lw=lw, zorder=3))
            elif radius:
                ax.add_patch(Circle((x, y), float(radius), fc=face, ec="0.12", lw=lw, zorder=4))
    for ob in trajectories:
        start, end = np.array(ob["start"]), np.array(ob["end"])
        ax.plot([start[0], end[0]], [start[1], end[1]], color="0.35", lw=.85, ls="--", zorder=2)
        ax.plot(end[0], end[1], marker="x", ms=3.5, color="0.2", zorder=5)
        label = "D" + ob["name"].split("_")[-1]
        ax.annotate(label, start, xytext=(5, 6), textcoords="offset points", fontsize=9,
                    bbox=dict(fc="white", ec="none", pad=.5), zorder=8)
    ax.set_xlim(-1.8, 18.8)
    ax.set_ylim(-1.8, 18.8)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("世界座標 x（m）", fontproperties=FONT, fontsize=12)
    ax.set_ylabel("世界座標 y（m）", fontproperties=FONT, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(fc="0.22", ec="black", label="牆體"),
               Patch(fc="0.78", ec="black", label="已知靜態障礙"),
               Patch(fc="white", ec="black", hatch="///", label="未知靜態障礙"),
               Line2D([], [], color="0.35", ls="--", label="動態障礙設定路段")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.12),
              ncol=2, prop=FONT10, frameon=False, columnspacing=1.5)
    export(fig, "dynamic_navigation_scene")


def comparison_plot(data, arrival_ids, clearance_ids, arrival):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.9))
    fig.subplots_adjust(left=.14, right=.98, bottom=.09, top=.87, hspace=.72)
    ax = axes[0]
    x = np.arange(3)
    values = np.array([arrival[key]["n_arrived"] for key in METHODS])
    failed = len(arrival_ids) - values
    ax.bar(x, values, width=.5, color="0.7", ec="black", linewidth=.75, label="到達")
    ax.bar(x, failed, bottom=values, width=.5, color="white", ec="black", linewidth=.75, hatch="///", label="未到達")
    for xx, n in zip(x, values):
        ax.text(xx, n/2, f"{n}", ha="center", va="center", fontsize=12)
    for xx, n, f in zip(x, values, failed):
        if f:
            ax.text(xx, n+f/2, f"{f}", ha="center", va="center", fontsize=11)
    ax.set_xticks(x, [LABELS[k] for k in METHODS], fontproperties=FONT, fontsize=12)
    ax.set_ylim(0, 94)
    ax.set_yticks([0, 20, 40, 60, 80])
    style(ax, "導航方法", "路線數", "(a) 共同路線之到達情形")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=2, frameon=False, prop=FONT10)
    ax.set_title("(a) 共同路線之到達情形", fontproperties=FONT, fontsize=13, pad=32)
    ax = axes[1]
    for key, ls in zip(METHODS, ["-", "--", ":"]):
        values = np.sort([float(data[key][i]["min_clearance_m"]) for i in clearance_ids])
        cdf_x = np.r_[-.62, values, .75]
        cdf_y = np.r_[0, 100*np.arange(1, len(values)+1)/len(values), 100]
        ax.step(cdf_x, cdf_y, where="post",
                color="0.1", ls=ls, lw=1.35, label=LABELS[key])
    ax.axvline(0, color="0.55", lw=.8)
    ax.set_xlim(-.62, .75)
    ax.set_ylim(0, 105)
    ax.set_xticks([-.6, -.3, 0, .3, .6])
    ax.set_yticks([0, 25, 50, 75, 100])
    style(ax, "每條路線之最小淨空（m）", "累積比例（%）", "(b) 最小淨空之經驗累積分布")
    ax.legend(loc="upper left", frameon=False, prop=FONT10)
    export(fig, "navigation_method_comparison")


def ablation_plot(data, ids, metrics):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.9))
    fig.subplots_adjust(left=.14, right=.98, bottom=.09, top=.94, hspace=.67)
    ax = axes[0]
    x = np.arange(4)
    values = [metrics[key]["n_overlap_routes"] for key in ABLATION]
    ax.bar(x, values, width=.5, color=["0.7", "white", "0.7", "white"],
           ec="black", linewidth=.75, hatch=["", "///", "", "///"])
    for xx, n in zip(x, values):
        ax.text(xx, n+.6, str(n), ha="center", va="bottom", fontsize=12)
    ax.set_ylim(0, 28)
    ax.set_yticks([0, 7, 14, 21, 28])
    ax.set_xticks(x, ["A", "B", "C", "D"])
    style(ax, "配置", "幾何重疊路線數", "(a) 避障模組與幾何重疊")
    ax = axes[1]
    times = [[float(data[key][i]["arrival_time_s"]) for i in ids] for key in ABLATION]
    ax.boxplot(times, positions=x, widths=.42, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="white", edgecolor="black", linewidth=.85),
               whiskerprops=dict(color="black", linewidth=.85),
               capprops=dict(color="black", linewidth=.85),
               medianprops=dict(color="black", linewidth=1.5))
    for xx, vals in zip(x, times):
        # Deterministic offsets separate samples visually, without changing data.
        offsets = np.linspace(-.12, .12, len(vals))
        ax.scatter(xx+offsets, vals, s=8, color="0.45", zorder=3, alpha=.65)
    ax.set_xticks(x, ["A", "B", "C", "D"])
    ax.set_ylim(0, max(max(v) for v in times)*1.08)
    style(ax, "配置", "到達時間（s）", "(b) 共同路線之到達時間分布")
    export(fig, "cbf_scan_shield_ablation")


def main():
    data, ids_a, ids_c, ids_b, arrival, clearance, ablation, excluded = extract_metrics()
    scene_plot()
    comparison_plot(data, ids_a, ids_c, arrival)
    ablation_plot(data, ids_b, ablation)
    source_paths = [ROOT / "evaluation/results" / key / "batch.csv" for key in sorted(data)]
    source_paths += [INPUT / "bigarena.sdf", INPUT / "dynamic_trajectories_bigarena_traffic.yaml",
                     ROOT / "evaluation/analyze.py", ROOT / "evaluation/record.sh",
                     ROOT / "evaluation/bags/archive_gmpc100/gmpc_cbf__scan_seed1/metadata.yaml",
                     ROOT / "evaluation/chain_three100.sh",
                     ROOT / "evaluation/chain_ablation2x2.sh", ROOT / "evaluation/summarise_three100.py",
                     ROOT / "evaluation/summarise_ablation2x2.py",
                     ROOT / "src/my_omnibot_description/config/nav2_baseline_mppi.yaml",
                     ROOT / "src/my_omnibot_description/config/nav2_baseline_rpp.yaml"]
    sources = [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths]
    manifest = {"section": "4.3", "new_simulations": False, "sources": sources,
                "arrival_common_routes": ids_a, "clearance_common_routes": ids_c,
                "clearance_exclusions": excluded, "ablation_common_routes": ids_b,
                "arrival_comparison": arrival, "clearance_comparison": clearance,
                "ablation": ablation,
                "definitions": {
                    "overlap_route": "A route with finite recorded min_clearance_m < 0. Count routes, not repeated collision_count events.",
                    "arrival_time": "Median among arriving routes for each method; arrival subsets can differ.",
                    "time_source": "Archived analyze.py uses rosbag receive timestamps, not message simulation stamps. Recorder timeout is 250 wall-clock seconds.",
                    "window": "Existing analysis: first global plan to first pose-based arrival; if no pose arrival, recording end. Saved success can also use controller arrival logs.",
                    "geometry_scope": "Existing static distance field and dynamic geometry approximation. Composite L shape is evaluated by its first rectangle in archived analyze.py; these statistics are not a complete contact-force or full-shape audit.",
                    "main_comparison_scope": "Configured navigation chains, different controller horizons and native motion capabilities. Reference detour disabled in archived GMPC parameters.",
                    "actuation_limits": "Equal execution smoothing limits on supported axes; RPP vy=0. MPPI internal ax_min=-1.5 remains different from execution decel=-6.25.",
                    "scene": "Archived SDF collision shapes at initial poses; dashed lines are configured obstacle travel segments, not measured robot trajectories."},
                "outputs": ["dynamic_navigation_scene.png", "navigation_method_comparison.png", "cbf_scan_shield_ablation.png"]}
    (OUT / "navigation_comparison_data.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"n_arrival":len(ids_a),"n_clearance":len(ids_c),"excluded":excluded,
                      "arrival":{k:arrival[k]["n_arrived"] for k in METHODS},
                      "clearance":clearance,"ablation":ablation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
