"""Build Chapter 4.2 figures from existing logs; no simulator or controller runs."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis_ch4_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FONT_PATH = ROOT / "預口試論文/overleaf/fonts/kaiu.ttf"
FONT = FontProperties(fname=str(FONT_PATH))
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                     "mathtext.fontset": "stix", "svg.fonttype": "path",
                     "savefig.facecolor": "white", "axes.unicode_minus": False})

IDENT = ROOT / "evaluation/results/wgmpc_arm_sp_ident_free4.json"
RUN = ROOT / "evaluation/runs/wgmpc_wg2_free4_182234"
BIAS = ROOT / "evaluation/runs/door_coopen_core_bench_3/analysis/door_coopen_core_bench_3_offset_shadow.json"
SOURCES = [IDENT, RUN / "sim/wb_run.json", RUN / "wg2_out.json", BIAS,
           ROOT / "evaluation/results/wgmpc_wg2_delay_check.yaml",
           ROOT / "evaluation/archive/door_coopen_core_bench_3_offset_shadow.py"]


def read(path):
    return json.loads(path.read_text())


def cn_label(ax, x=None, y=None, title=None):
    if x:
        ax.set_xlabel(x, fontproperties=FONT, fontsize=12)
    if y:
        ax.set_ylabel(y, fontproperties=FONT, fontsize=12)
    if title:
        ax.set_title(title, fontproperties=FONT, fontsize=13, pad=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.85", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)


def legend(ax, **kwargs):
    fp = FontProperties(fname=str(FONT_PATH), size=10)
    ax.legend(prop=fp, frameon=False, **kwargs)


def export(fig, name):
    fig.canvas.draw()
    # Log locators also create labels beyond the view; those are not drawn by
    # Axis. Exclude them from the bounds check explicitly.
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


def model_metrics():
    ident, sim, solver = read(IDENT), read(RUN / "sim/wb_run.json"), read(RUN / "wg2_out.json")
    cols = {key: i for i, key in enumerate(sim["log_cols"])}
    data = np.asarray(sim["log"], float)
    times = data[:, cols["t"]]
    q = np.column_stack([data[:, cols[f"joint{i}_act"]] for i in range(1, 7)])
    s = np.column_stack([data[:, cols[f"joint{i}_sp"]] for i in range(1, 7)])
    window = np.flatnonzero((times >= ident["task_window_s"][0]) & (times <= ident["task_window_s"][1]))
    left, right = int(window[0]), int(window[-1])
    finite = np.isfinite(s).all(axis=1) & np.isfinite(q).all(axis=1)
    assert finite[left:right + 1].all()
    dtp = float(np.median(np.diff(times[window])))
    n_train = int(ident["n_train"])
    x = s[left:right] - q[left:right]
    y = q[left + 1:right + 1] - q[left:right]
    fitted = np.array([np.linalg.lstsq(np.column_stack([x[:n_train, j], np.ones(n_train)]),
                                     y[:n_train, j], rcond=None)[0] for j in range(6)])
    alpha, bias = fitted[:, 0], fitted[:, 1]
    assert np.allclose(alpha, ident["alpha"], atol=1e-13, rtol=0)
    assert np.allclose(bias, ident["bias_rad"], atol=1e-13, rtol=0)
    val_left = left + n_train

    def starts_for(horizon_steps):
        starts = []
        for rec in solver["log"]:
            if not rec.get("published"):
                continue
            i = int(np.argmin(np.abs(times - rec["sim_t"])))
            if (i >= max(val_left, left + 1) and i + horizon_steps < right
                    and abs(times[i] - rec["sim_t"]) <= dtp
                    and finite[i - 1:i + horizon_steps + 2].all()):
                starts.append(i)
        return np.array(sorted(set(starts)), dtype=int)

    def evaluate(starts, steps):
        # Same interval convention as the archived identification script:
        # s[i] acts on q[i] -> q[i+1]; the command already written into s[i]
        # belongs to the first physical interval of the direct-rate baseline.
        command_integral = s[starts + steps - 1] - s[starts - 1]
        direct = q[starts] + command_integral
        tracked = q[starts].copy()
        averaged = q[starts].copy()
        s_average = s[starts].copy()
        u_average = command_integral / (steps * dtp)
        for k in range(steps):
            tracked += alpha * (s[starts + k] - tracked) + bias
            averaged += alpha * (s_average - averaged) + bias
            s_average += dtp * u_average
        target = q[starts + steps]
        rms = lambda pred: np.sqrt(np.mean((1000 * (pred - target)) ** 2, axis=0)).tolist()
        return {"direct": rms(direct), "augmented_logged_setpoints": rms(tracked),
                "augmented_future_interval_mean_input": rms(averaged)}

    starts50 = starts_for(5)
    rmse50 = evaluate(starts50, 5)
    tags = {"direct": "理想速度積分（現行核心）",
            "augmented_logged_setpoints": "設定點追蹤｜設定點取實錄",
            "augmented_future_interval_mean_input": "設定點追蹤｜事後視窗平均固定輸入"}
    for key, suffix in tags.items():
        assert np.allclose(rmse50[key], ident["rmse_mrad"][f"對齊控制週期｜{suffix}"], atol=1e-10)
    steps = [1, 5, 10, 15, 25]
    common = starts_for(max(steps))
    curves = [{"horizon_ms": round(h * dtp * 1000), "rmse_mrad": evaluate(common, h)} for h in steps]
    err1 = q[val_left + 1:right + 1] - (q[val_left:right] + alpha * (s[val_left:right] - q[val_left:right]) + bias)
    return {"physical_dt_s": dtp, "alpha": alpha.tolist(), "bias_rad_per_step": bias.tolist(),
            "tau_s": (-dtp / np.log(1 - alpha)).tolist(),
            "equivalent_offset_mrad": (1000 * bias / alpha).tolist(),
            "n_train": n_train, "n_valid": right - val_left,
            "validation_one_physics_step_rmse_mrad": (1000 * np.sqrt(np.mean(err1 ** 2, axis=0))).tolist(),
            "n_50ms_control_aligned_starts": len(starts50), "rmse_50ms_mrad": rmse50,
            "n_common_horizon_starts": len(common), "horizon_curves": curves,
            "scope": "Offline conditional replay with recorded future setpoints/commands. The mean-input variant also uses future observations."}


def model_plot(metrics):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.7))
    fig.subplots_adjust(left=.14, right=.98, bottom=.09, top=.87, hspace=.72)
    joints = np.arange(1, 7)
    series = [("direct", "直接速度積分", "o", "-", "0.1"),
              ("augmented_logged_setpoints", "增廣模型（實錄設定點）", "s", "--", "0.1"),
              ("augmented_future_interval_mean_input", "增廣模型（區間平均輸入）", "^", ":", "0.45")]
    ax = axes[0]
    short_labels = ["直接速度積分", "增廣（設定點）", "增廣（平均輸入）"]
    for (key, label, marker, style, color), short_label in zip(series, short_labels):
        ax.plot(joints, metrics["rmse_50ms_mrad"][key], marker=marker,
                linestyle=style, color=color, linewidth=1.2, markersize=5,
                markerfacecolor="white", label=short_label)
    ax.set_yscale("log")
    ax.set_xticks(joints)
    ax.set_ylim(.0005, 100)
    cn_label(ax, "關節編號", "預測 RMSE（mrad）", "(a) 50 ms 逐軸預測誤差")
    ax.set_title("(a) 50 ms 逐軸預測誤差", fontproperties=FONT, fontsize=13, pad=34)
    legend(ax, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3,
           columnspacing=1.0, handletextpad=.5, borderaxespad=0)
    ax = axes[1]
    horizons = [row["horizon_ms"] for row in metrics["horizon_curves"]]
    for key, label, marker, style, color in series[:2]:
        values = [max(row["rmse_mrad"][key]) for row in metrics["horizon_curves"]]
        ax.plot(horizons, values, marker=marker, linestyle=style, color=color,
                linewidth=1.2, markersize=5, markerfacecolor="white", label=label)
    ax.set_yscale("log")
    ax.set_xticks(horizons)
    cn_label(ax, "預測時間長度（ms）", "最大逐軸 RMSE（mrad）", "(b) 預測時間長度與誤差")
    legend(ax, loc="lower right")
    export(fig, "arm_model_prediction_validation")


def bias_plot():
    data = read(BIAS)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.7))
    fig.subplots_adjust(left=.14, right=.98, bottom=.09, top=.93, hspace=.64)
    ax = axes[0]
    joints = np.arange(1, 7)
    for values, shift, label, color, hatch in [
            (data["overall"]["rms_res_frozen"], -.18, "固定偏置", "0.7", ""),
            (data["overall"]["rms_res_shadow"], .18, "區間更新", "white", "///")]:
        ax.bar(joints + shift, 1000 * np.asarray(values), width=.34,
               color=color, edgecolor="black", linewidth=.7, hatch=hatch, label=label)
    ax.set_yscale("log")
    ax.set_xticks(joints)
    cn_label(ax, "關節編號", "區間殘差 RMS（mrad）", "(a) 更新前估計之逐軸殘差")
    legend(ax, loc="upper right")
    ax = axes[1]
    centers = [np.mean(row["q_bin_deg"]) for row in data["bins"]]
    for suffix, label, style, marker in [("frozen", "固定偏置", "-", "o"),
                                         ("shadow", "區間更新", "--", "s")]:
        ax.plot(centers, [row[f"rms_vz_{suffix}_mm"] for row in data["bins"]],
                color="0.1", linestyle=style, marker=marker, markerfacecolor="white",
                markersize=5, linewidth=1.2, label=label)
    ax.set_xticks(np.arange(0, 46, 5))
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 1.55)
    cn_label(ax, "門角（度）", "鉛直投影殘差 RMS（mm）", "(b) 各門角區間之鉛直投影殘差")
    legend(ax, loc="upper left")
    export(fig, "interval_bias_prediction_residual")
    return {"n_intervals": data["n_intervals"], "overall": data["overall"],
            "bins": data["bins"], "initial_equivalent_offset_rad": data["d0_frozen"],
            "filter_tau_s": .5, "absolute_offset_bound_rad": .05,
            "scope": "Fixed measured trajectory; previous-interval estimate predicts each endpoint before updating. Jacobian projection is not measured TCP tracking error."}


def main():
    metrics = model_metrics()
    model_plot(metrics)
    bias = bias_plot()
    sources = [{"path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in SOURCES]
    out = {"chapter": "4.2", "sources": sources, "model_validation": metrics,
           "bias_replay": bias,
           "outputs": ["arm_model_prediction_validation.png", "interval_bias_prediction_residual.png"],
           "font": "DFKai-SB / 標楷體", "new_simulations": False,
           "validation": "Refit matches archived alpha and bias; 50 ms metrics match all six archived control-aligned joint RMSEs."}
    (OUT / "execution_model_validation_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rmse_50ms_mrad": metrics["rmse_50ms_mrad"],
                      "horizon_max_joint_rmse_mrad": [{"ms": r["horizon_ms"],
                      "direct": max(r["rmse_mrad"]["direct"]),
                      "augmented": max(r["rmse_mrad"]["augmented_logged_setpoints"])} for r in metrics["horizon_curves"]],
                      "common_horizon_starts": metrics["n_common_horizon_starts"],
                      "bias_overall": bias["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
