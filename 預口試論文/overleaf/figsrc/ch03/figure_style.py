"""Print-size typography and caption text for the original thesis diagrams."""
import re
import os
import tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis_ch03_mpl"))
import matplotlib

PRINT_WIDTH_IN = 5.8
MIN_PRINT_FONT_PT = 11.0

CAPTIONS = {
    "base_pose_error": "底盤追蹤之相對幾何關係：(a) 世界座標中的參考與實際底盤位姿；(b) 以參考底盤為基準的相對平移與航向。",
    "wholebody_motion_coupling": "底盤側移時之全身運動學耦合：(a) 底盤移動前；(b) 底盤側移後，手臂調整構型以維持相同末端目標。",
    "arm_setpoint_tracking": "手臂設定點與實際關節位置之追蹤關係：(a) 設定點固定；(b) 關節持續追蹤；(c) 關節逐漸接近設定點。虛線與實線分別表示位置設定點與實際關節位置。",
    "dynamic_obstacle_prediction": "同一未來時刻下的候選運動比較：(a) 目前觀測；(b) 維持原速度；(c) 降低速度；(d) 側向避讓。",
    "visual_operation_reference": "操作目標導引之底盤選位：(a) 選位前的機器人與目標位置；(b) 選位後，手臂接近目標。",
    "grasp_state_transition": "抓持狀態之轉移：(a) 張爪接近目標；(b) 閉爪並檢查抓持證據；(c) 證據持續達標且取得必要的連接確認後，進入抓持狀態。",
    "door_operation_sequence": "門扇操作之全身構型：(a) 抓持初始門角；(b) 門扇開啟後，底盤與手臂配合把手運動。",
    "drawer_operation_sequence": "抽屜操作之全身構型：(a) 抽屜初始位置；(b) 抽屜沿滑軌拉出後，底盤與手臂配合調整。",
    "base_relative_carrying": "相對底盤目標下之持物運送：(a) 底盤移動前；(b) 底盤平移後，維持相同的底盤相對末端構型。",
    "omni_wheel_velocity_feasible_set": r"四輪十字配置之速度可行域：(a) 固定旋轉速度下的 $v_x$--$v_y$ 平移可行域；(b) $v_y=0$ 時的 $v_x$--$\omega$ 剖面。虛線表示各軸獨立速度框，灰色表示輪速限制下的可行區域。",
    "mobile_approach_distance_profile": r"圓柱接近階段之距離規則：(a) 平移逐軸速度上限；(b) 手臂距離展開進度與後段併線門檻。圖中為設計曲線，實際進度另受時間與狀態限制。",
    "tcp_target_compensation": "末端原始幾何目標與有界命令目標補償；完成條件依原始幾何目標判定。",
}


def prepare(fig):
    """Keep labels readable when the complete figure is printed 14.7 cm wide."""
    fig.canvas.draw()
    minimum = MIN_PRINT_FONT_PT * fig.get_figwidth() / PRINT_WIDTH_IN
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        # The panel meaning belongs in the LaTeX caption, not a second title.
        match = re.match(r"^\(([a-z])\)\s+.+", artist.get_text(), re.S)
        if match:
            artist.set_text(f"({match.group(1)})")
        artist.set_fontsize(max(artist.get_fontsize(), minimum))
    fig.canvas.draw()
    return dict(
        print_width_cm=round(PRINT_WIDTH_IN * 2.54, 2),
        minimum_print_font_pt=MIN_PRINT_FONT_PT,
        explanatory_footer=False,
        panel_descriptions_in_caption=True,
    )
