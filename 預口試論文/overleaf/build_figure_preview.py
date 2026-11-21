#!/usr/bin/env python3
"""Build an independent figure-selection PDF without changing source images."""
from pathlib import Path
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'masterthesis_preview_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / '第一二章_插圖預覽.pdf'
FONT = FontProperties(fname=str(ROOT / 'fonts/kaiu.ttf'))
plt.rcParams.update({'pdf.fonttype': 42, 'axes.unicode_minus': False})


def make_page(pdf, index, title, position, parts, source):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
    fig.text(.05, .94, title, fontproperties=FONT, fontsize=17, va='top')
    fig.text(.05, .885, '建議位置：' + position, fontproperties=FONT, fontsize=11, va='top')
    fig.add_artist(plt.Line2D([.05, .95], [.86, .86], transform=fig.transFigure, color='black', linewidth=.65))
    for rel, bounds, label in parts:
        ax = fig.add_axes(bounds)
        ax.imshow(plt.imread(ROOT / rel))
        ax.set_axis_off()
        if label:
            fig.text(bounds[0]+bounds[2]/2, .14, label,
                     fontproperties=FONT, fontsize=11, ha='center', va='center')
    fig.text(.05, .055, '來源：' + source, fontproperties=FONT, fontsize=9.2, va='bottom')
    fig.text(.95, .055, f'{index} / 9', fontproperties=FONT, fontsize=9.2, va='bottom', ha='right')
    pdf.savefig(fig)
    plt.close(fig)


with PdfPages(OUTPUT, metadata={'Title': '第一、二章插圖選用預覽', 'Author': '本研究插圖整理',
                                'Subject': '獨立選圖預覽；非論文編譯頁面'}) as pdf:
    make_page(pdf, 1, '移動機械臂之工業與室內應用平台', '1.1.1 自主移動機械臂之發展與應用', [
        ('figsrc/ch01/kuka_kmr_iisy_application.jpg', [.05,.20,.44,.61], '(a) KUKA KMR iisy'),
        ('figsrc/ch01/stretch4_platform_views.jpg', [.52,.20,.43,.61], '(b) Hello Robot Stretch 4')],
        'KUKA、Hello Robot 官方產品頁；照片保留原貌。')
    make_page(pdf, 2, '關節式物件操作之研究實例', '1.1.2 動態環境下之導航與操作需求', [
        ('figsrc/ch01/mittal2022_articulated_fig1.png', [.13,.115,.74,.70], '')],
        'Mittal 等人，IROS 2022，Fig. 1；含模擬與實機場景。')
    make_page(pdf, 3, '移動中抓取與放置之連續動作示意', '1.3.2 全身模型預測控制與移動中操作', [
        ('figsrc/ch01/burgess2023_motm_banner.png', [.05,.19,.90,.62], '')],
        'Burgess-Limerick 作者專案 Manipulation On-The-Move；模擬動作示意。')
    make_page(pdf, 4, '移動機械臂整機模型', '2.1 移動機械臂平台', [
        ('figsrc/ch02/mobile_manipulator_model.png', [.20,.115,.60,.70], '')],
        '本專案既有 URDF 視覺網格離線渲染。')
    make_page(pdf, 5, '手腕相機與末端夾爪配置', '2.2.1 雷射掃描與深度相機', [
        ('figsrc/ch02/wrist_camera_gripper_model.png', [.20,.115,.60,.70], '')],
        '本專案既有模型近照。')
    make_page(pdf, 6, 'Isaac Sim 門與桌邊取物區域', '2.3.2 Isaac Sim 物理模擬與場景建置', [
        ('figsrc/ch02/isaac_door_area.png', [.05,.20,.44,.61], '(a) 門與通行區域'),
        ('figsrc/ch02/isaac_table_area.png', [.52,.20,.43,.61], '(b) 桌邊取物區域')],
        'm8house_d2_unified_19 既有紀錄重播影格；原始字幕保留。')
    make_page(pdf, 7, '導航、視覺與操作之整體控制架構設計', '2.4 導航、視覺與操作之系統整合，節首', [
        ('figsrc/ch02/system_control_architecture.png', [.05,.115,.90,.70], '')],
        '本研究自行繪製；黑白線條、標楷體。')
    make_page(pdf, 8, '底盤幾何模型預測導航架構', '2.4.1 全域路徑規劃與導航參考', [
        ('figsrc/ch02/navigation_architecture.png', [.05,.115,.90,.70], '')],
        '本研究自行繪製；沿用第五次報告布局，更新命令處理鏈。')
    make_page(pdf, 9, '門把手之 RGB 與對應深度觀測', '2.4.2 視覺感知與操作目標資訊', [
        ('figsrc/ch02/door_handle_rgb.png', [.05,.20,.44,.61], '(a) RGB 原始影像'),
        ('figsrc/ch02/door_handle_depth.png', [.52,.20,.43,.61], '(b) 同一影格之深度資料')],
        'm8doorcap_1 靜態開發集；深度單位 m，白色為無效深度。')
print(OUTPUT)
