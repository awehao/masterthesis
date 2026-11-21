#!/usr/bin/env python3
"""Draw matching black-and-white thesis architecture diagrams.

Run with Python 3 and Matplotlib. Output files are written beside this script.
The navigation layout follows the fifth progress report's phase-one diagram.
No simulation, inference, or raster-image modification is performed.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "masterthesis_architecture_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = Path(__file__).resolve().parent
INK = "#000000"
RULE = "#000000"


# Use the exact DFKai-SB font supplied with the thesis template. Do not silently
# substitute another Kai family. The override supports a relocated source file.
FONT_PATH = Path(os.environ.get(
    "THESIS_KAIU_FONT", str(OUT.parents[1] / "fonts" / "kaiu.ttf")
)).resolve()
if not FONT_PATH.is_file():
    raise FileNotFoundError(
        "DFKai-SB is required: provide fonts/kaiu.ttf or set THESIS_KAIU_FONT"
    )
KAI_FONT = FontProperties(fname=str(FONT_PATH))
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})


class Diagram:
    def __init__(self, name, width, height, inches):
        self.name = name
        self.fig, self.ax = plt.subplots(figsize=inches)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set(xlim=(0, width), ylim=(0, height))
        self.ax.set_axis_off()
        self.nodes = {}
        self.frames = {}
        self.text_checks = []

    def text(self, x, y, value, size=8.0, bold=False, **kwargs):
        artist = self.ax.text(x, y, value, ha="center", va="center",
                              fontproperties=KAI_FONT, fontsize=size,
                              color=INK, zorder=5, **kwargs)
        # DFKai-SB has one regular face. A light vector stroke emphasizes titles
        # while retaining the same Kai letterforms throughout the figure.
        if bold:
            artist.set_path_effects([
                patheffects.withStroke(linewidth=0.15, foreground=INK)
            ])
        return artist

    def frame(self, key, x, y, width, height, linewidth=1.0):
        rect = Rectangle((x-width/2, y-height/2), width, height,
                         linewidth=linewidth, edgecolor=RULE,
                         facecolor="white", zorder=3)
        self.ax.add_patch(rect)
        self.nodes[key] = (x, y, width, height)
        self.frames[key] = rect
        return rect

    def node_text(self, key, y, value, size=9.0, bold=False):
        x, _, _, _ = self.nodes[key]
        rect = self.frames[key]
        artist = self.text(x, y, value, size=size, bold=bold)
        self.text_checks.append((key, rect, artist))
        return artist

    def node(self, key, x, y, width, height, title, details=(), linewidth=1.0):
        rect = self.frame(key, x, y, width, height, linewidth)
        lines = [title] + list(details)
        step = 0.40
        offset = step * (len(lines)-1)/2
        for i, line in enumerate(lines):
            artist = self.text(x, y+offset-i*step, line,
                               size=9.8 if i == 0 else 9.0,
                               bold=i == 0)
            self.text_checks.append((key, rect, artist))
        return rect

    def port(self, key, side, offset=0):
        x, y, w, h = self.nodes[key]
        return {"left": (x-w/2, y+offset), "right": (x+w/2, y+offset),
                "top": (x+offset, y+h/2), "bottom": (x+offset, y-h/2)}[side]

    def arrow(self, points, feedback=False):
        xs, ys = zip(*points)
        style = (0, (4, 2.5)) if feedback else "solid"
        if len(points) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=RULE, linewidth=0.95,
                         linestyle=style, solid_joinstyle="miter", zorder=2)
        self.ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=8.0,
            linewidth=0.95, color=RULE, linestyle=style,
            shrinkA=0, shrinkB=1.0, zorder=2))

    def junction(self, x, y):
        self.ax.plot(x, y, "o", color=RULE, markersize=2.4, zorder=4)

    def export(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        failures = []
        for key, rect, artist in self.text_checks:
            inner = rect.get_window_extent(renderer)
            box = artist.get_window_extent(renderer)
            if not (inner.x0+2 < box.x0 and box.x1 < inner.x1-2
                    and inner.y0+1 < box.y0 and box.y1 < inner.y1-1):
                failures.append({"node": key, "text": artist.get_text()})
        if failures:
            raise RuntimeError("Text exceeds node bounds: " + json.dumps(failures, ensure_ascii=False))
        for ext in ["pdf", "svg", "png"]:
            self.fig.savefig(OUT/f"{self.name}.{ext}", dpi=300,
                             metadata={"Creator": "Matplotlib; masterthesis architecture source"})
        plt.close(self.fig)
        print(self.name + ": PDF, SVG, PNG generated; labels fit their nodes")


def draw_system():
    d = Diagram("system_control_architecture", 16.8, 8.7, (7.2, 7.2*8.7/16.8))
    # Three parallel inputs, one shared controller, and two execution branches.
    d.node("scan", 1.6, 6.95, 2.65, 1.15, "雷射掃描", ["環境原始量測"])
    d.node("camera", 1.6, 4.85, 2.65, 1.15, "RGB-D 影像", ["底盤與手腕相機"])
    d.node("task", 1.6, 2.90, 2.65, 1.50, "地圖與任務資訊", ["區域與物件類型", "幾何與機構先驗"])
    d.node("obstacles", 5.25, 6.95, 3.65, 1.50, "障礙物感知與預測", ["掃描處理、追蹤與分類", "障礙位置與運動資訊"])
    d.node("vision", 5.25, 4.85, 3.65, 1.50, "視覺感知與目標定位", ["目標分割與三維定位", "座標轉換與有效性檢查"])
    d.node("references", 5.25, 2.90, 3.65, 1.50, "任務與參考生成", ["導航路徑與末端目標", "任務階段與物件約束"])
    d.frame("controller", 9.55, 5.0, 3.80, 5.40, linewidth=1.25)
    d.node_text("controller", 7.22, "全身 GMPC", size=11.0, bold=True)
    d.node_text("controller", 6.78, "幾何模型預測控制", size=9.3)
    d.node("prediction", 9.55, 5.58, 3.32, 1.55, "協同預測模型", ["底盤與手臂協同預測", "命令執行動態與延遲"], linewidth=0.75)
    d.node("optimization", 9.55, 3.60, 3.32, 1.75, "共同最佳化問題", ["底盤與末端追蹤", "避障與機器人運動限制", "物件機構限制"], linewidth=0.75)
    d.node("base", 14.0, 6.95, 3.55, 1.50, "底盤命令處理", ["速度平滑與掃描防護", "輪速與輪加速度限制"])
    d.node("arm", 14.0, 4.00, 3.55, 1.50, "手臂命令處理", ["關節位置設定點積分", "位置命令追蹤"])
    d.node("robot", 11.7125, 1.10, 8.125, 1.25, "移動機械臂執行端", ["命令接收、底盤／手臂／夾爪與物理模擬"])
    d.node("state", 5.25, 1.10, 3.65, 1.35, "狀態與執行資訊", ["位姿、關節與設定點", "實際命令與在途命令"])
    d.arrow([d.port("scan", "right"), d.port("obstacles", "left")])
    d.arrow([d.port("camera", "right"), d.port("vision", "left")])
    d.arrow([d.port("task", "right"), d.port("references", "left")])
    d.arrow([d.port("vision", "bottom"), d.port("references", "top")])
    d.arrow([d.port("obstacles", "right"), d.port("controller", "left", 1.95)])
    d.arrow([d.port("references", "right"), d.port("controller", "left", -2.10)])
    d.arrow([d.port("controller", "right", 1.95), d.port("base", "left")])
    d.arrow([d.port("controller", "right", -1.00), d.port("arm", "left")])
    d.arrow([d.port("base", "right"), (16.20, 6.95), (16.20, 1.10), d.port("robot", "right")])
    d.arrow([d.port("arm", "bottom"), d.port("robot", "top", 2.2875)])
    d.arrow([(3.15, 6.95), (3.15, 8.15), (14.0, 8.15), d.port("base", "top")])
    d.junction(3.15, 6.95)
    d.text(8.4, 8.42, "原始雷射掃描（直接送入底盤防護）", size=8.7)
    d.arrow([d.port("robot", "left"), d.port("state", "right")], feedback=True)
    # Records are aggregated from the execution interface and the simulator;
    # the physical robot itself is not assumed to measure in-transit commands.
    d.arrow([d.port("state", "top"), (5.25, 1.98), (9.55, 1.98), d.port("controller", "bottom")], feedback=True)
    d.export()


def draw_navigation():
    d = Diagram("navigation_architecture", 16.8, 9.4, (7.2, 7.2*9.4/16.8))
    # Keep the fifth report's planning / perception / localization rows.
    d.node("map", 1.6, 7.65, 2.65, 1.15, "目標點與環境地圖", ["目標與靜態空間"])
    d.node("planner", 5.25, 7.65, 3.65, 1.90, "路徑規劃與參考生成", ["Nav2 全域路徑規劃", "路徑取樣與航向生成", "預測時域內之參考序列"])
    d.node("scan", 1.6, 4.90, 2.65, 1.15, "2D LiDAR", ["環境雷射掃描"])
    d.node("obstacles", 5.25, 4.90, 3.65, 2.05, "雷射感知與障礙物追蹤", ["自體遮罩／地圖相減／群集", "KF 追蹤與動靜分流", "表面點與障礙運動預測"])
    d.node("odom", 1.6, 2.05, 2.65, 1.15, "里程計", ["底盤運動狀態"])
    d.node("localization", 5.25, 2.05, 3.65, 1.50, "定位：AMCL＋EKF", ["結合地圖、雷射與里程計", "底盤位姿與速度估測"])
    d.frame("controller", 9.55, 4.90, 3.80, 6.20, linewidth=1.25)
    d.node_text("controller", 7.50, "SE(2) GMPC", size=11.0, bold=True)
    d.node_text("controller", 7.04, "幾何模型預測控制", size=9.3)
    d.node_text("controller", 6.51, "參考追蹤與速度增量平滑")
    # CBF is part of the same optimization, rather than a serial post-filter.
    d.node("cbf", 9.55, 4.90, 3.32, 1.80, "全時域預測式 CBF", ["動態位置預測與靜態障礙", "避障條件納入同一 QP"], linewidth=0.75)
    d.node_text("controller", 3.46, "速度與加速度限制")
    d.node_text("controller", 3.00, "輪速與輪加速度限制")
    d.node_text("controller", 2.36, "滾動時域最佳化求解")
    d.node("smoother", 14.0, 7.05, 3.55, 1.15, "速度平滑器", ["Velocity Smoother"])
    d.node("shield", 14.0, 5.20, 3.55, 1.45, "近距離安全層", ["Raw-scan Safety Shield", "掃描防護與輪級條件"])
    d.node("guard", 14.0, 3.35, 3.55, 1.15, "輪級命令限制器", ["輪速與輪加速度限制"])
    d.node("robot", 14.0, 1.55, 3.55, 1.15, "全向底盤執行", ["最終底盤速度命令"])
    d.arrow([d.port("map", "right"), d.port("planner", "left")])
    d.arrow([d.port("planner", "right"), d.port("controller", "left", 2.75)])
    d.arrow([d.port("scan", "right"), d.port("obstacles", "left")])
    d.arrow([d.port("obstacles", "right"), d.port("controller", "left")])
    d.arrow([d.port("odom", "right"), d.port("localization", "left")])
    d.arrow([d.port("localization", "right"), (7.35, 2.05), (7.35, 2.70), d.port("controller", "left", -2.20)])
    d.arrow([d.port("controller", "right", 2.15), d.port("smoother", "left")])
    d.arrow([d.port("smoother", "bottom"), d.port("shield", "top")])
    d.arrow([d.port("shield", "bottom"), d.port("guard", "top")])
    d.arrow([d.port("guard", "bottom"), d.port("robot", "top")])
    # Raw-scan bypass follows the outside of the original report's layout.
    d.arrow([d.port("scan", "left"), (0.08, 4.90), (0.08, 8.87), (16.12, 8.87), (16.12, 5.50), d.port("shield", "right", 0.30)])
    d.text(8.4, 9.15, "原始雷射掃描（直接送入安全層）", size=8.7)
    d.arrow([d.port("guard", "right"), (16.12, 3.35), (16.12, 4.85), d.port("shield", "right", -0.35)], feedback=True)
    d.text(16.43, 4.08, "最終命令", size=7.8, rotation=90)
    # A single bottom return path retains the report's readable closed loop.
    d.arrow([d.port("robot", "bottom"), (14.0, 0.52), (1.6, 0.52), d.port("odom", "bottom")], feedback=True)
    d.text(7.35, 0.79, "底盤運動與里程計回授", size=8.5)
    d.export()


if __name__ == "__main__":
    draw_system()
    draw_navigation()
