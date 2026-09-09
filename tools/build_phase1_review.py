#!/usr/bin/env python3
"""Render a source-grounded, single-page phase-one review (SVG/PNG/PDF)."""
from pathlib import Path
from html import escape
import ctypes as C
from ctypes.util import find_library

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '第五次進度報告素材'
OUT.mkdir(exist_ok=True)
STEM = OUT / '第一階段回顧_單頁系統流程圖'
W, H = 1920, 1080
NAVY, INK, MUTED = '#163451', '#23394D', '#607387'
BLUE, TEAL, ORANGE = '#2879B9', '#148579', '#CB6B27'
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
     '<title>第一階段回顧：動態環境下的全向底盤預測導航</title>',
     '<desc>上方回顧前四次報告，下方呈現規劃、感知、定位、GMPC與CBF、平滑器、原始雷射安全層及底盤回授。</desc>',
     '<defs>']
for name, color in [('blue', BLUE), ('teal', TEAL), ('orange', ORANGE), ('gray', MUTED)]:
    s.append(f'<marker id="{name}" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L10,5 L0,10 Z" fill="{color}"/></marker>')
s.append('</defs>')

def rect(x, y, w, h, fill, stroke='none', radius=16):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

def text(x, y, value, size=24, color=INK, weight=400, anchor='start'):
    s.append(f'<text x="{x}" y="{y}" font-family="Noto Sans CJK TC, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')

def arrow(points, color=BLUE, marker='blue', dashed=False):
    pts = ' '.join(f'{x},{y}' for x, y in points)
    dash = ' stroke-dasharray="10 7"' if dashed else ''
    s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"{dash} marker-end="url(#{marker})"/>')

def label(x, y, value, size=20, color=MUTED):
    text(x, y, value, size, color)

def box(x, y, w, h, title, lines, color=BLUE, fill='#F1F7FC', title_size=27):
    rect(x, y, w, h, fill, color)
    text(x + 22, y + 40, title, title_size, color, 700)
    for i, line in enumerate(lines):
        text(x + 22, y + 76 + i * 32, line, 22)

rect(0, 0, W, H, '#FFFFFF', radius=0)
rect(0, 0, 14, H, NAVY, radius=0)
text(64, 58, '第 5 次進度報告  /  進度回顧', 22, MUTED)
text(64, 113, '第一階段：動態環境下的全向底盤預測導航', 43, NAVY, 700)
text(1854, 58, 'ROS 2 Jazzy × Gazebo Harmonic', 21, MUTED, anchor='end')

cards = [
    ('01', '平台與導航基礎', 'ROS 2／Gazebo、TF、AMCL', '底盤模型與 Nav2 導航基準', BLUE),
    ('02', '預測控制核心', 'SE(2) GMPC＋KF 障礙物預測', '整合全時域 CBF，真值場景驗證', BLUE),
    ('03', '感測與控制整合', 'omni_bot、LiDAR 追蹤、定位', '靜態 CBF、控制平滑與比較實驗', TEAL),
    ('04', '系統補強與安全層', '表面點／淨位移分流、輪系限制', 'Raw-scan Shield、隨機路線驗證', ORANGE),
]
for i, (num, title, l1, l2, c) in enumerate(cards):
    x = 64 + 454 * i
    rect(x, 145, 428, 130, '#F5F8FB')
    text(x + 18, 182, num, 28, c, 700)
    text(x + 70, 182, title, 26, NAVY, 700)
    text(x + 18, 222, l1, 21)
    text(x + 18, 254, l2, 21)

text(64, 320, '第一階段完成的系統資料流', 27, NAVY, 700)

# Draw connections before boxes so endpoints sit cleanly on the borders.
arrow([(344, 445), (420, 445)])
arrow([(820, 445), (930, 445)])
label(842, 429, '參考路徑', 19)
arrow([(344, 634), (420, 634)], TEAL, 'teal')
arrow([(820, 624), (930, 624)], TEAL, 'teal')
label(841, 607, '位置／速度', 18, TEAL)
arrow([(344, 828), (420, 828)])
arrow([(820, 828), (873, 828), (873, 764), (930, 764)])
label(830, 862, '底盤位姿', 19)
arrow([(1300, 476), (1400, 476)])
label(1314, 457, '速度命令', 18)
arrow([(1550, 524), (1550, 588)])
arrow([(1550, 720), (1550, 784)], ORANGE, 'orange')
label(1567, 762, '/cmd_vel', 20, ORANGE)

# The shield's scan input deliberately bypasses perception and classification.
# Route around the left outside edge so this never crosses the planner input.
# Keep the caption above the uninterrupted line, with rounded elbow geometry.
s.append(f'<path d="M 64 586 H 48 Q 36 586 36 574 V 385 '
         f'Q 36 373 48 373 H 1764 Q 1776 373 1776 385 '
         f'V 638 Q 1776 650 1764 650 H 1703" fill="none" '
         f'stroke="{ORANGE}" stroke-width="3" stroke-dasharray="10 7" '
         f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#orange)"/>')
text(888, 359, '原始 /scan：直接送入 Shield，不經動靜分類', 22, ORANGE, 600)

# Closed-loop odometry feedback.
arrow([(1550, 896), (1550, 929), (204, 929), (204, 882)], MUTED, 'gray', True)
rect(740, 911, 360, 36, '#FFFFFF', radius=0)
text(758, 937, '底盤運動 → 里程計回授', 22, MUTED)

box(64, 393, 280, 104, '目標點＋環境地圖', ['導航任務與靜態空間'], title_size=25)
box(64, 554, 280, 118, '2D LiDAR', ['環境掃描 /scan'], TEAL, '#EFF9F6')
box(64, 774, 280, 108, '里程計', ['底盤運動狀態'], title_size=27)

box(420, 393, 400, 104, 'Nav2 全域規劃', ['Costmap → 靜態繞行路徑'])
box(420, 548, 400, 176, '雷射感知與障礙物追蹤', ['自體遮罩／地圖相減 → 群集', 'KF 追蹤 → 淨位移動靜分流', '表面點表示 → 障礙物位置／速度'], TEAL, '#EFF9F6', 26)
box(420, 772, 400, 124, '定位：AMCL＋EKF', ['結合地圖、雷射與里程計', 'Beam-skip／融合與取值修正'], title_size=26)

text(1115, 403, '預測避障', 23, BLUE, 700, 'middle')
rect(930, 414, 370, 394, '#EDF4FC', BLUE)
text(955, 456, 'SE(2) GMPC', 32, BLUE, 700)
text(955, 494, '幾何模型預測控制', 26, NAVY, 600)
text(955, 537, '參考追蹤＋速度增量平滑', 23)
rect(950, 566, 330, 126, '#FFFFFF', '#BBD3E8', 12)
text(967, 601, '全時域預測式 CBF', 25, BLUE, 700)
text(967, 638, '動態位置外推＋靜態牆面', 22)
text(967, 673, '安全約束納入同一 QP', 22)
text(955, 733, '納入輪系速度／加速度限制', 22)
text(955, 777, '20 Hz 閉迴路求解', 24, BLUE, 600)

box(1400, 428, 300, 96, '速度平滑器', ['velocity smoother'], title_size=27)
text(1530, 574, '近距離安全保護', 22, ORANGE, 700, 'end')
box(1400, 588, 300, 132, '近距離安全層', ['Raw-scan Safety Shield', '限制朝障礙物的接近速度'], ORANGE, '#FFF5EC', 27)
box(1400, 784, 300, 112, '全向底盤執行', ['omni_bot 模擬平台'], title_size=27)

rect(64, 975, 1790, 70, NAVY)
text(88, 1005, '階段成果', 23, '#FFFFFF', 700)
text(222, 1005, '完成感知—規劃—控制—安全的導航閉迴路，並以隨機路線與 MPPI／RPP 對照驗證。', 24, '#FFFFFF')
text(222, 1032, '第一階段範圍：底盤導航模擬驗證；作為後續底盤與手臂協同控制的基礎。', 20, '#CCDAE8')
s.append('</svg>')
svg = '\n'.join(s)
STEM.with_suffix('.svg').write_text(svg)
# Call installed native libraries directly; no optional Python GI cairo bridge.
rsvg = C.CDLL(find_library('rsvg-2'))
cairo = C.CDLL(find_library('cairo'))
gobject = C.CDLL(find_library('gobject-2.0'))
def bind(lib, name, args, result):
    f = getattr(lib, name)
    f.argtypes, f.restype = args, result
    return f
ptr, dbl, integer = C.c_void_p, C.c_double, C.c_int
new_svg = bind(rsvg, 'rsvg_handle_new_from_data', [C.c_char_p, C.c_size_t, ptr], ptr)
class Rectangle(C.Structure):
    _fields_ = [(n, dbl) for n in ['x', 'y', 'width', 'height']]
render = bind(rsvg, 'rsvg_handle_render_document', [ptr, ptr, C.POINTER(Rectangle), ptr], integer)
new_png = bind(cairo, 'cairo_image_surface_create', [integer, integer, integer], ptr)
new_pdf = bind(cairo, 'cairo_pdf_surface_create', [C.c_char_p, dbl, dbl], ptr)
context = bind(cairo, 'cairo_create', [ptr], ptr)
scale = bind(cairo, 'cairo_scale', [ptr, dbl, dbl], None)
write_png = bind(cairo, 'cairo_surface_write_to_png', [ptr, C.c_char_p], integer)
finish = bind(cairo, 'cairo_surface_finish', [ptr], None)
surface_status = bind(cairo, 'cairo_surface_status', [ptr], integer)
destroy_context = bind(cairo, 'cairo_destroy', [ptr], None)
destroy_surface = bind(cairo, 'cairo_surface_destroy', [ptr], None)
unref = bind(gobject, 'g_object_unref', [ptr], None)
data = svg.encode()
handle = new_svg(data, len(data), None)
assert handle, 'SVG parsing failed'
viewport = Rectangle(0, 0, W, H)
for fmt, factor in [('png', 2), ('pdf', .5)]:
    path = str(STEM.with_suffix('.' + fmt)).encode()
    surface = new_png(0, W * 2, H * 2) if fmt == 'png' else new_pdf(path, W / 2, H / 2)
    ctx = context(surface)
    scale(ctx, factor, factor)
    assert render(handle, ctx, C.byref(viewport), None), 'SVG rendering failed'
    if fmt == 'png':
        assert write_png(surface, path) == 0, 'PNG writing failed'
    finish(surface)
    assert surface_status(surface) == 0, 'Cairo surface failed'
    destroy_context(ctx)
    destroy_surface(surface)
unref(handle)
print(STEM)
