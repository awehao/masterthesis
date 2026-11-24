"""Original, print-readable figures for the condensed chapter; no simulator runs."""
from pathlib import Path
import os
import tempfile
import json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis_ch3_condensed_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BUNDLE = ROOT / "預口試論文/第三章_精簡重整版"
OUT = BUNDLE / "figsrc/ch03"
OUT.mkdir(parents=True, exist_ok=True)
FONT = FontProperties(fname=str(ROOT / "預口試論文/overleaf/fonts/kaiu.ttf"))
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "stix",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "svg.fonttype": "path",
    "font.size": 11,
})
INK = "#161616"
GRAY = "#747474"
LIGHT = "#eeeeee"
BOXES = []
RECORDS = []


def txt(ax, x, y, value, size=11, chinese=False, **kw):
    opts = dict(ha="center", va="center", fontsize=size, color=INK, linespacing=1.15, zorder=20)
    if chinese:
        opts["fontproperties"] = FONT
    opts.update(kw)
    return ax.text(x, y, value, **opts)


def box(ax, x, y, w, h, value, size=11, fill="white", chinese=False):
    r = Rectangle((x-w/2, y-h/2), w, h, ec=INK, fc=fill, lw=.85, zorder=5)
    ax.add_patch(r)
    t = txt(ax, x, y, value, size=size, chinese=chinese)
    BOXES.append((r, t))
    return r


def arrow(ax, a, b, dashed=False, rad=0):
    p = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
                        lw=.85, color=INK, ls="--" if dashed else "-", zorder=8)
    ax.add_patch(p)


def path(ax, pts, **kw):
    a = np.asarray(pts)
    opts = dict(color=INK, lw=.85, zorder=7)
    opts.update(kw)
    ax.plot(a[:, 0], a[:, 1], **opts)


def export(fig, name, source):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        if min(b.x0, b.y0) < -1 or b.x1 > w+1 or b.y1 > h+1:
            raise RuntimeError(f"{name}: clipped text: {t.get_text()}")
    errors = []
    for r, t in BOXES:
        if r.figure is not fig:
            continue
        rb = r.get_window_extent(renderer)
        tb = t.get_window_extent(renderer)
        if not (rb.x0+1 < tb.x0 and tb.x1 < rb.x1-1 and rb.y0+1 < tb.y0 and tb.y1 < rb.y1-1):
            errors.append(t.get_text())
    if errors:
        raise RuntimeError(f"{name}: text outside boxes: {errors}")
    fig.savefig(OUT / f"{name}.png", dpi=400)
    # Editable native vector master; the thesis itself uses PNG only.
    fig.savefig(OUT / f"{name}.svg")
    RECORDS.append(dict(file=f"{name}.png", source=source,
                        original_drawing=True, print_width_inches=5.8,
                        layout_bounds_check="passed"))
    plt.close(fig)
    print(name, flush=True)


def rotation_comparison():
    fig = plt.figure(figsize=(5.8, 2.05))
    ax = fig.add_axes([.025, .16, .39, .76])
    ax.set_aspect("equal")
    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=GRAY, lw=.9)
    ax.plot([-1.18, 1.18], [0, 0], color=GRAY, lw=.5)
    ax.plot([0, 0], [-1.18, 1.18], color=GRAY, lw=.5)
    for angle, ls in [(np.deg2rad(170), "-"), (np.deg2rad(190), "--")]:
        pt = np.array([np.cos(angle), np.sin(angle)])
        path(ax, [[0, 0], pt], ls=ls, lw=1.2)
        ax.plot(*pt, marker="o", color=INK, ms=4)
    theta = np.deg2rad(np.linspace(152, 208, 60))
    path(ax, np.c_[1.12*np.cos(theta), 1.12*np.sin(theta)], lw=1.25)
    arrow(ax, (1.12*np.cos(theta[-3]), 1.12*np.sin(theta[-3])),
          (1.12*np.cos(theta[-1]), 1.12*np.sin(theta[-1])))
    txt(ax, 0, -1.38, "(a)", 12)
    ax.set(xlim=(-1.5, 1.4), ylim=(-1.58, 1.4))
    ax.axis("off")
    ax = fig.add_axes([.56, .24, .405, .62])
    u = np.linspace(0, 1, 200)
    true = np.deg2rad(150+60*u)
    wrapped = (true+np.pi) % (2*np.pi)-np.pi
    cut = np.where(np.abs(np.diff(wrapped)) > np.pi)[0][0]
    ax.plot(u[:cut+1], wrapped[:cut+1], color=INK, lw=1.4)
    ax.plot(u[cut+1:], wrapped[cut+1:], color=INK, lw=1.4)
    ax.plot([.5, .5], [-np.pi, np.pi], color=GRAY, ls=":", lw=.8)
    ax.set(xlim=(0, 1), ylim=(-3.7, 3.7), xticks=[],
           yticks=[-np.pi, 0, np.pi], yticklabels=[r"$-\pi$", "$0$", r"$\pi$"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)
    txt(ax, .5, -.18, "(b)", size=12, transform=ax.transAxes)
    export(fig, "rotation_representation_redrawn",
           "Original SO(2)/wrapped-angle illustration based on Tang et al. (2024).")


def kalman_cycle():
    fig = plt.figure(figsize=(5.8, 1.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    box(ax, 2.30, 1.65, 4.0, 1.45,
        "Prediction\n"+r"$\hat{x}^{-}=F_o\hat{x}^{+}_{\rm prev}$"+"\n"+
        r"$P^{-}=F_oP^{+}_{\rm prev}F_o^{\mathsf{T}}+Q$", 11)
    box(ax, 7.55, 1.65, 4.3, 1.45,
        "Measurement update\n"+r"$r=y-H_o\hat{x}^{-}$"+"\n"+
        r"$\hat{x}^{+}=\hat{x}^{-}+Kr$", 11)
    arrow(ax, (4.30, 1.65), (5.40, 1.65))
    txt(ax, 4.85, 2.05, r"$\hat{x}^{-},P^{-}$", 11)
    txt(ax, 7.55, 2.80, r"Measurement $y$", 11)
    arrow(ax, (7.55, 2.58), (7.55, 2.39))
    path(ax, [(7.55, .925), (7.55, .38), (2.30, .38)])
    arrow(ax, (2.30, .38), (2.30, .925))
    txt(ax, 4.90, .43, r"$\hat{x}^{+},P^{+}$",
        11, bbox=dict(fc="white", ec="none", pad=2))
    export(fig, "kalman_tracking_redrawn",
           "Original prediction/update cycle; Kalman (1960), Welch and Bishop (2004).")


def robot(ax, p, ghost=False):
    p = np.asarray(p)
    ax.add_patch(Circle(p, .30, ec=GRAY if ghost else INK, fc="white",
                        lw=.85, ls=":" if ghost else "-", zorder=6))
    if ghost:
        return
    for angle in np.arange(4)*np.pi/2:
        n = np.array([np.cos(angle), np.sin(angle)])
        t = np.array([-n[1], n[0]])
        c = p+.245*n
        verts = [c+i*.025*n+j*.060*t for i, j in [(-1,-1), (1,-1), (1,1), (-1,1)]]
        ax.add_patch(Polygon(verts, ec=INK, fc=LIGHT, lw=.5, zorder=7))
    arrow(ax, p+[-.1, 0], p+[.12, 0])


def pedestrian(ax, p):
    x, y = p
    ax.add_patch(Circle((x, y), .25, ec=GRAY, fc=LIGHT, lw=.8, zorder=8))
    ax.add_patch(Circle((x, y+.10), .065, ec=INK, fc=INK, zorder=9))
    path(ax, [(x, y+.02), (x, y-.075)], lw=2, zorder=9)
    path(ax, [(x-.095, y-.01), (x+.095, y-.01)], lw=1.4, zorder=9)
    path(ax, [(x-.07, y-.17), (x, y-.075), (x+.07, y-.17)], lw=1.1, zorder=9)


def dynamic_row():
    fig = plt.figure(figsize=(5.8, 1.65))
    start = np.array([.4, 0.])
    old = np.array([1.6, 1.4])
    future = np.array([1.6, 0.])
    positions = [None, future, np.array([.78, 0.]), np.array([1.6, -.72])]
    for i in range(4):
        ax = fig.add_axes([.008+i*.25, .10, .236, .88])
        ax.set_aspect("equal")
        ax.set(xlim=(-.02, 2.10), ylim=(-1.14, 1.78))
        ax.axis("off")
        path(ax, [(-.02, 0), (2.10, 0)], color="#bbbbbb", ls=":", lw=.6, zorder=1)
        if i == 0:
            robot(ax, start)
            pedestrian(ax, old)
            arrow(ax, (.80, 0), (1.28, 0))
            arrow(ax, (1.6, 1.01), (1.6, .58))
        else:
            robot(ax, start, ghost=True)
            path(ax, [old, future], color=GRAY, ls=":", lw=.7, zorder=2)
            arrow(ax, (1.6, 1.16), (1.6, .36), dashed=True)
            pts = [start, positions[i]] if i != 3 else [start, [1.02, -.34], positions[i]]
            path(ax, pts, lw=1.0, zorder=3)
            robot(ax, positions[i])
            pedestrian(ax, future)
            if i != 1:
                v = future-positions[i]
                v /= np.linalg.norm(v)
                a = positions[i]+.30*v
                b = future-.25*v
                ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<->", mutation_scale=6,
                                            color=INK, lw=.7, zorder=12))
        fig.text(.125+i*.25, .055, f"({chr(97+i)})", fontsize=11, ha="center", va="center")
    export(fig, "dynamic_obstacle_prediction",
           "Original four-panel schematic: current observation and three candidates at one future time.")


def mask_architecture():
    """Inference architecture of the actual torchvision R50-FPN-v2 configuration."""
    fig = plt.figure(figsize=(5.8, 5.50))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 12), ylim=(0, 12))
    ax.axis("off")
    # Backbone: ResNet-50 bottleneck stage counts [3, 4, 6, 3].
    txt(ax, .30, 11.70, "(a)", 11, ha="left")
    box(ax, 1.15, 10.85, 2.12, .88, "RGB\n"+r"$3\times480\times640$", 10.7)
    box(ax, 3.25, 10.85, 1.55, .88, "Stem\n"+r"$7\times7$; pool", 10.7)
    arrow(ax, (2.21, 10.85), (2.475, 10.85))
    centers = [5.08, 7.08, 9.08, 11.08]
    for i, (x, n, channels) in enumerate(zip(centers, [3, 4, 6, 3], [256, 512, 1024, 2048])):
        box(ax, x, 10.85, 1.78, 1.10, f"C{i+2}\n{n} blocks\n{channels} ch", 10.7)
        if i == 0:
            arrow(ax, (4.025, 10.85), (x-.89, 10.85))
        else:
            arrow(ax, (centers[i-1]+.89, 10.85), (x-.89, 10.85))
    # Lateral transforms, top-down additions and output convolutions.
    # Bottleneck detail fits in the unused space beside the FPN lateral layers.
    txt(ax, 2.04, 10.17, "Residual block", 10.7)
    for x, kernel in [(.70, r"$1\times1$"), (1.90, r"$3\times3$"), (3.10, r"$1\times1$")]:
        box(ax, x, 9.35, .92, .60, kernel, 10.7)
    arrow(ax, (.05, 9.35), (.24, 9.35))
    arrow(ax, (1.16, 9.35), (1.44, 9.35))
    arrow(ax, (2.36, 9.35), (2.64, 9.35))
    ax.add_patch(Circle((3.94, 9.35), .17, ec=INK, fc="white", lw=.8, zorder=10))
    txt(ax, 3.94, 9.35, "+", 10.7)
    arrow(ax, (3.56, 9.35), (3.77, 9.35))
    path(ax, [(.12, 9.35), (.12, 9.85), (3.94, 9.85)])
    arrow(ax, (3.94, 9.85), (3.94, 9.52))
    txt(ax, 2.00, 9.85, "skip / proj.", 10.7,
        bbox=dict(fc="white", ec="none", pad=1.3))
    txt(ax, 2.03, 8.55, "FPN: "+r"$3\times3$ Conv", 10.7)
    for i, x in enumerate(centers):
        box(ax, x, 9.25, 1.42, .52, r"$1\times1$", 11)
        arrow(ax, (x, 10.30), (x, 9.51))
        arrow(ax, (x, 8.99), (x, 8.68))
        ax.add_patch(Circle((x, 8.49), .19, ec=INK, fc="white", lw=.8, zorder=10))
        txt(ax, x, 8.49, "+", 11)
        box(ax, x, 7.71, 1.74, .74, f"P{i+2}\n256 ch", 10.7, fill=LIGHT)
        arrow(ax, (x, 8.30), (x, 8.08))
        if i < 3:
            arrow(ax, (centers[i+1]-.20, 8.49), (x+.20, 8.49))
            txt(ax, x+1.0, 8.79, r"$\uparrow2$", 10.7)
    box(ax, 2.00, 7.71, 1.78, .74, "P6\npool(P5)", 10.7, fill=LIGHT)
    # P6 is max-pooled directly from P5, not part of the top-down additions.
    path(ax, [(11.08, 7.34), (11.08, 7.02), (2.00, 7.02)])
    arrow(ax, (2.00, 7.02), (2.00, 7.34))
    # Distinct inputs keep proposal coordinates separate from feature tensors.
    txt(ax, .30, 6.55, "(b)", 11, ha="left")
    box(ax, 1.25, 5.93, 2.0, .80, "P2--P6\nfeatures", 11, fill=LIGHT)
    box(ax, 4.00, 5.93, 2.9, .80, "RPN\n"+r"$2\times(3\times3)$ Conv", 10.7)
    box(ax, 7.48, 5.93, 3.15, .80, r"$1\times1$ Conv"+"\nObjectness / box", 10.7)
    box(ax, 10.65, 5.93, 2.05, .80, "NMS\nProposals", 11)
    arrow(ax, (2.25, 5.93), (2.55, 5.93))
    arrow(ax, (5.45, 5.93), (5.905, 5.93))
    arrow(ax, (9.055, 5.93), (9.625, 5.93))
    # Detection branch. The v2 head has four Conv-BN-ReLU blocks and one FC.
    txt(ax, .30, 4.96, "(c)", 11, ha="left")
    box(ax, 1.20, 4.13, 1.95, 1.05, "P2--P5\nfeatures", 11, fill=LIGHT)
    box(ax, 3.73, 4.13, 2.45, 1.05, "RoIAlign\n"+r"$256\times7\times7$", 10.7)
    box(ax, 6.58, 4.13, 2.55, 1.05, r"$4\times$ Conv"+"\nFC 1024", 11)
    box(ax, 10.05, 4.13, 3.42, 1.05, "Class scores: 7\nBox offsets: "+r"$7\times4$", 10.7)
    arrow(ax, (2.175, 4.13), (2.505, 4.13))
    arrow(ax, (4.955, 4.13), (5.305, 4.13))
    arrow(ax, (7.855, 4.13), (8.34, 4.13))
    # Proposal coordinates route above the branch to its RoIAlign input.
    path(ax, [(10.65, 5.53), (10.65, 5.14), (3.73, 5.14)])
    arrow(ax, (3.73, 5.14), (3.73, 4.655))
    # Mask branch uses detection boxes at inference, not raw proposal boxes.
    txt(ax, .30, 3.16, "(d)", 11, ha="left")
    box(ax, 1.20, 2.28, 1.95, 1.05, "P2--P5\nfeatures", 11, fill=LIGHT)
    box(ax, 3.73, 2.28, 2.45, 1.05, "RoIAlign\n"+r"$256\times14\times14$", 10.7)
    box(ax, 6.58, 2.28, 2.55, 1.05, r"$4\times$ Conv"+"\n"+r"$2\times2$ deconv", 10.7)
    box(ax, 10.05, 2.28, 3.42, 1.05, r"$1\times1$ Conv"+"\n"+r"$7\times28\times28$", 10.7)
    arrow(ax, (2.175, 2.28), (2.505, 2.28))
    arrow(ax, (4.955, 2.28), (5.305, 2.28))
    arrow(ax, (7.855, 2.28), (8.34, 2.28))
    path(ax, [(10.05, 3.605), (10.05, 3.27), (3.73, 3.27)])
    arrow(ax, (3.73, 3.27), (3.73, 2.805))
    txt(ax, 6.52, 3.31, "Detection boxes",
        10.7, bbox=dict(fc="white", ec="none", pad=1.5))
    box(ax, 6.30, .66, 10.75, .74,
        "Select class mask "+r"$\rightarrow$"+" sigmoid "+r"$\rightarrow$"+
        " resize to image "+r"$\rightarrow$"+" binary mask", 10.7)
    path(ax, [(10.05, 1.755), (10.05, 1.28), (6.30, 1.28)])
    arrow(ax, (6.30, 1.28), (6.30, 1.03))
    export(fig, "mask_rcnn_architecture",
           "Original inference diagram checked against evaluation/house_seg.py and torchvision "
           "maskrcnn_resnet50_fpn_v2 source. Six foreground classes plus background.")


if __name__ == "__main__":
    rotation_comparison()
    kalman_cycle()
    dynamic_row()
    mask_architecture()
    (BUNDLE / "新繪圖片核對.json").write_text(
        json.dumps(RECORDS, ensure_ascii=False, indent=2), encoding="utf-8")
