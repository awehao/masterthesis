#!/usr/bin/env python3
"""Build editable vector figures and 4K PNGs for the fifth progress report."""
from pathlib import Path
from html import escape
import ctypes as C
from ctypes.util import find_library
import zipfile
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '第五次進度報告素材' / '本次進度圖表'
W, H = 1920, 1080
NAVY, INK, MUTED = '#163451', '#23394D', '#607387'
BLUE, TEAL, ORANGE = '#2879B9', '#148579', '#CB6B27'
LIGHT, LINE = '#F3F7FB', '#DCE5EE'


class Figure:
    def __init__(self, number, title, subtitle):
        self.s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                  f'<title>{escape(title)}</title>', f'<desc>{escape(subtitle)}</desc>', '<defs>']
        for color in (BLUE, TEAL, ORANGE, MUTED):
            self.s.append(f'<marker id="a{color[1:]}" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L12 6 L0 12 Z" fill="{color}"/></marker>')
        self.s.append('</defs>')
        self.rect(0, 0, W, H, 'white', radius=0)
        self.rect(0, 0, 14, H, NAVY, radius=0)
        self.text(64, 55, '第五次進度報告  /  本次進度與成果', 22, MUTED)
        self.text(1856, 55, number, 24, TEAL, 700, 'end')
        self.text(64, 122, title, 47, NAVY, 700)
        self.text(64, 174, subtitle, 27, MUTED)

    def rect(self, x, y, w, h, fill=LIGHT, stroke='none', radius=16):
        self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def text(self, x, y, value, size=28, color=INK, weight=400, anchor='start'):
        self.s.append(f'<text x="{x}" y="{y}" font-family="Noto Sans CJK TC, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def lines(self, x, y, lines, size=28, color=INK, step=44, weight=400):
        for i, value in enumerate(lines):
            self.text(x, y + i * step, value, size, color, weight)

    def line(self, pts, color=BLUE, width=4, arrow=False, dashed=False):
        props = f' marker-end="url(#a{color[1:]})"' if arrow else ''
        props += ' stroke-dasharray="10 9"' if dashed else ''
        self.s.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{props}/>')

    def circle(self, x, y, r, fill='white', stroke=BLUE, width=3):
        self.s.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def box(self, x, y, w, h, title, lines=(), color=BLUE, fill=LIGHT, size=30):
        self.rect(x, y, w, h, fill, LINE)
        self.rect(x, y, 6, h, color, radius=0)
        self.text(x+24, y+45, title, size, color, 700)
        self.lines(x+24, y+91, lines, 27, step=41)

    def footer(self, main, note):
        self.rect(64, 932, 1792, 98, NAVY)
        self.text(88, 973, main, 29, 'white', 600)
        self.text(88, 1009, note, 22, '#D3E1EE')
        self.text(1856, 1060, '資料截至 2026-09-10', 18, MUTED, anchor='end')

    def table(self, x, y, widths, headers, rows, row_h=78, sizes=None):
        total = sum(widths)
        self.rect(x, y, total, 66, NAVY, radius=0)
        pos = x
        for w, h in zip(widths, headers):
            self.text(pos+22, y+44, h, 27, 'white', 600)
            pos += w
        for i, row in enumerate(rows):
            yy = y+66+i*row_h
            self.rect(x, yy, total, row_h, '#F0F7F6' if i == 0 else ('#F3F7FB' if i%2==0 else '#FFFFFF'), radius=0)
            self.line([(x, yy+row_h), (x+total, yy+row_h)], LINE, 1)
            xx = x
            for j, (w, cell) in enumerate(zip(widths, row)):
                lines = cell.split('\n')
                size = sizes[j] if sizes else 28
                baseline = yy + row_h/2 - (len(lines)-1)*18 + size*.35
                for k, value in enumerate(lines):
                    self.text(xx+22, baseline+k*36, value, size, TEAL if j==1 else INK, 600 if j==1 else 400)
                xx += w

    def save(self, stem):
        svg = '\n'.join(self.s + ['</svg>'])
        path = OUT / stem
        path.with_suffix('.svg').write_text(svg)
        render_files(svg, path)
        return path


def bind(lib, name, args, result):
    f = getattr(lib, name)
    f.argtypes, f.restype = args, result
    return f


def render_files(svg, path):
    rsvg, cairo, gob = [C.CDLL(find_library(n)) for n in ('rsvg-2', 'cairo', 'gobject-2.0')]
    ptr, dbl, integer = C.c_void_p, C.c_double, C.c_int
    new = bind(rsvg, 'rsvg_handle_new_from_data', [C.c_char_p, C.c_size_t, ptr], ptr)
    class Rect(C.Structure):
        _fields_ = [(n, dbl) for n in ('x', 'y', 'width', 'height')]
    render = bind(rsvg, 'rsvg_handle_render_document', [ptr, ptr, C.POINTER(Rect), ptr], integer)
    png = bind(cairo, 'cairo_image_surface_create', [integer, integer, integer], ptr)
    pdf = bind(cairo, 'cairo_pdf_surface_create', [C.c_char_p, dbl, dbl], ptr)
    ctx_new = bind(cairo, 'cairo_create', [ptr], ptr)
    scale = bind(cairo, 'cairo_scale', [ptr, dbl, dbl], None)
    write = bind(cairo, 'cairo_surface_write_to_png', [ptr, C.c_char_p], integer)
    finish = bind(cairo, 'cairo_surface_finish', [ptr], None)
    status = bind(cairo, 'cairo_surface_status', [ptr], integer)
    destroy = bind(cairo, 'cairo_destroy', [ptr], None)
    destroy_s = bind(cairo, 'cairo_surface_destroy', [ptr], None)
    unref = bind(gob, 'g_object_unref', [ptr], None)
    data = svg.encode()
    handle = new(data, len(data), None)
    assert handle
    for fmt, factor in [('png', 2), ('pdf', .5)]:
        target = str(path.with_suffix('.'+fmt)).encode()
        surface = png(0, W*2, H*2) if fmt=='png' else pdf(target, W/2, H/2)
        ctx = ctx_new(surface)
        scale(ctx, factor, factor)
        assert render(handle, ctx, C.byref(Rect(0, 0, W, H)), None)
        if fmt == 'png':
            assert write(surface, target) == 0
        finish(surface)
        assert status(surface) == 0
        destroy(ctx)
        destroy_s(surface)
    unref(handle)


def model():
    f = Figure('01 / 進度', '全身模型：底盤與手臂形成同一條運動鏈', '全向底盤 3 自由度＋Lite 6 手臂 6 自由度，共同決定工具中心點 TCP 的位姿。')
    f.rect(64, 216, 850, 668, LIGHT)
    f.text(94, 263, '9 自由度移動機械臂', 31, NAVY, 700)
    f.text(882, 263, '幾何示意・不依比例', 20, MUTED, anchor='end')
    # A schematic side view; no claim of CAD-accurate mounting or pose.
    f.line([(145, 752), (830, 752)], LINE, 3)
    f.rect(211, 628, 405, 104, '#DCE9F4', BLUE, 38)
    for x in (265, 559):
        f.rect(x-33, 699, 66, 46, NAVY, radius=12)
    f.rect(380, 590, 85, 45, '#B9D1E7', BLUE, 8)
    joints = [(420,590), (420,522), (339,403), (481,325), (579,377), (657,377)]
    f.line(joints, '#B6CCC6', 39)
    f.line(joints, TEAL, 25)
    for i, (x,y) in enumerate(joints):
        f.circle(x,y,18, 'white', TEAL, 4)
        dx, dy = (-43, 2) if i in (0,1,2) else (-6,-33)
        f.text(x+dx,y+dy,f'J{i+1}',20,TEAL,600)
    f.rect(673, 354, 39, 48, NAVY, radius=6)
    f.line([(707,359),(739,359),(739,369)], NAVY, 8)
    f.line([(707,397),(739,397),(739,387)], NAVY, 8)
    f.circle(740,378,6, ORANGE, ORANGE)
    f.line([(740,378),(805,378)], ORANGE, 3, True)
    f.text(742,341,'TCP',26,ORANGE,700)
    f.text(558,479,'Lite 6：6 DOF',29,TEAL,700)
    f.line([(558,490),(527,490),(481,355)],TEAL,2)
    f.text(94,568,'全向底盤',29,BLUE,700)
    f.line([(232,570),(273,570),(303,628)],BLUE,2)
    f.line([(371,813),(508,813)], BLUE, 4, True)
    f.line([(371,813),(319,769)], BLUE, 4, True)
    f.text(519,823,'x',26,BLUE,700)
    f.text(293,772,'y',26,BLUE,700)
    f.text(602,820,'平移 x、y＋旋轉 θ',26,BLUE,600)
    f.box(958,216,898,178,'單一全身運動鏈', ['q = [ x, y, θ, q₁, …, q₆ ]ᵀ', '全身 Jacobian 同時納入底盤與手臂的速度貢獻'])
    f.box(958,418,898,208,'模型與模擬交叉驗證', ['FK 對獨立運動學實作', 'Jacobian 對數值微分', 'Gazebo 核對關節方向、零位與移動時末端位姿'],TEAL)
    f.box(958,650,898,234,'模型一致性修正', ['由 URDF 讀取幾何、掛載點與工具座標', '修正車身高度偏差，同步更新預抓取參考', '運動學模型供求解；模擬模型供 Gazebo 執行'])
    f.footer('本期進度：建立可共同解算底盤與手臂動作的 9 自由度模型。', '這是模型與模擬一致性驗證；尚未量測實機末端定位精度。')
    return f.save('01_全身模型與運動學示意圖')


def safety():
    f = Figure('02 / 進度','安全範圍：從底盤圓盤擴充至連桿表面','隨關節姿態更新表面距離與接近速度限制，並處理手臂自體回波。')
    f.rect(64,216,1050,668,LIGHT)
    f.text(96,263,'連桿表面與安全距離示意',31,NAVY,700)
    f.text(1078,263,'幾何示意・不依比例',20,MUTED,anchor='end')
    f.rect(893,330,157,424,'#FBEADC',ORANGE,6)
    f.text(971,799,'已知障礙',27,ORANGE,600,'middle')
    # Link capsule in local coordinates; samples follow its surface.
    f.s.append('<g transform="translate(224 631) rotate(-29)">')
    f.rect(0,-53,472,106,'#DEEEE9',TEAL,53)
    for x,y in [(30,-45),(86,-53),(145,-53),(204,-53),(263,-53),(322,-53),(381,-53),(435,-44),(464,-15),(464,15),(435,44),(381,53),(322,53),(263,53),(204,53),(145,53),(86,53),(30,45),(7,15),(7,-15)]:
        f.circle(x,y,5,TEAL,TEAL,1)
    f.s.append('</g>')
    f.circle(269,607,23,'white',TEAL,4)
    f.text(133,733,'連桿碰撞網格',29,TEAL,700)
    f.text(133,779,'表面取樣＋涵蓋半徑 ρ',27,INK)
    f.line([(403,725),(420,688),(465,553)],TEAL,2)
    f.circle(643,420,10,ORANGE,ORANGE,2)
    f.line([(643,420),(887,420)],ORANGE,3,True)
    f.text(765,395,'表面距離 d',28,ORANGE,600,'middle')
    f.line([(643,432),(751,487)],TEAL,4,True)
    f.text(758,517,'接近速度',26,TEAL,600)
    f.line([(643,405),(643,324)],BLUE,4,True)
    f.text(518,316,'切向運動',25,BLUE)
    f.text(609,611,'表面點 Jacobian',26,NAVY,600)
    f.text(609,653,'限制法向接近速度',26,INK)
    f.box(1150,216,706,193,'① 動態自體濾波',['依姿態移除手臂自身回波','遮擋區域仍視為未知空間'],BLUE)
    f.box(1150,432,706,220,'② 全身安全與運動限制',['距離餘裕考量延遲與制動需求','共同限制底盤與關節速度','納入關節位置、加速度與 jerk'],TEAL)
    f.box(1150,675,706,209,'③ 資料失效處理',['距離整批過期 → 零命令','約束資料遭截斷 → 零命令','以實際輸出歷史與時間間隔更新'],ORANGE)
    f.footer('本期進度：將幾何安全與運動限制接入全身控制路徑。','連桿距離來自已知場景／模擬資料；尚未接入腕部深度感知，示意不代表完整安全證明。')
    return f.save('02_連桿安全與自體濾波示意圖')


def control():
    f = Figure('03 / 進度','協同控制：在安全可行範圍內共同完成末端任務','有障礙案例揭露「安全停住，但末端未到達」；建立整合約束的單步全身 QP 基線。')
    f.rect(64,216,1792,277,LIGHT)
    f.text(94,262,'原架構｜先求任務速度，再由安全層修正',31,NAVY,700)
    for x,w,title,lines in [(96,350,'末端任務',['位置＋工具朝向']),(511,385,'任務速度解算',['先分配底盤／手臂動作']),(962,377,'安全濾波',['修正不安全的命令']),(1405,418,'執行與回授',['底盤＋手臂'])]:
        f.box(x,296,w,128,title,lines,BLUE,size=28)
    for a,b in [(446,511),(896,962),(1339,1405)]:
        f.line([(a+8,360),(b-12,360)],BLUE,3,True)
    f.text(96,464,'觀察：底盤受阻後，手臂未接手；末端停在距目標 225.9 mm 處。',27,ORANGE,600)
    f.rect(64,518,1792,357,'#EDF7F4')
    f.text(94,565,'本期基線｜任務與安全約束整合求解',31,TEAL,700)
    f.box(96,595,353,157,'末端任務＋目前狀態',['TCP 目標、底盤位姿','關節姿態與障礙距離'],TEAL,size=27)
    f.box(514,595,636,184,'單一步全身 QP',['共同考慮任務誤差、構形偏好','連桿安全與底盤／關節運動限制','在可行速度集合內分配動作'],TEAL,size=29)
    f.box(1215,595,285,157,'下游安全濾波',['保留於閉迴路','屬於基線的一部分'],TEAL,size=26)
    f.box(1565,595,258,157,'執行與回授',['底盤＋手臂','9 自由度速度'],TEAL,size=27)
    for a,b in [(449,514),(1150,1215),(1500,1565)]:
        f.line([(a+8,673),(b-12,673)],TEAL,3,True)
    f.text(96,836,'主要設定：構形正則權重 μ = 0.03；同一有障礙案例重複 3 次皆完成預抓取。',28,TEAL,600)
    f.footer('研究發現：安全可行集合內的任務最佳化，有助於重新分配底盤與手臂動作。','到達結果同時涉及架構整合與構形權重調整；目前為單步 QP，全身多步 W-GMPC 尚待開發。')
    return f.save('03_協同控制架構對照圖')


def platforms():
    f = Figure('04 / 進度','模擬平台與驗證：建立可重複量測的控制基線','Gazebo 驗證全身預抓取；Isaac Sim 建立導航與影像介面，並修正模擬時間基準。')
    f.rect(64,216,871,668,LIGHT)
    f.rect(963,216,893,668,'#EDF7F4')
    f.text(96,270,'Gazebo｜全身任務與控制排程',33,BLUE,700)
    f.text(997,270,'Isaac Sim｜導航介面與時間驗證',33,TEAL,700)
    f.box(96,305,807,175,'有障礙協同預抓取',['底盤＋手臂同動，追蹤 TCP 位置與姿態','凍結單步 QP、下游濾波器與任務設定'],BLUE, 'white')
    f.line([(499,490),(499,530)],BLUE,4,True)
    f.box(96,546,807,190,'控制迴圈排程修正',['原：等待完整 50 ms 後才運算','改：運算後只等待距截止時間的剩餘時間','加速度限制使用實測發布間隔'],BLUE,'white')
    f.text(126,806,'發布間隔中位',27,MUTED)
    f.text(126,851,'61.19 ms → 50.0 ms',40,BLUE,700)
    f.box(997,305,827,175,'介面接通',['模型、scan、odom、TF、速度命令','底盤 RGB 影像供觀看與記錄'],TEAL,'white')
    f.line([(1410,490),(1410,530)],TEAL,4,True)
    f.box(997,546,827,190,'物理時間與 ROS 時鐘對齊',['統一 physics_dt 與 rendering_dt','由物理世界時間發布 /clock','依模擬時間安排感測與渲染更新'],TEAL,'white')
    f.text(1027,806,'修正後重跑',27,MUTED)
    f.text(1027,851,'前方／左側／後方：皆到達',35,TEAL,700)
    f.footer('本期進度：全身控制基線已凍結；Isaac 初步導航與時間基準驗證完成。','Isaac 三方向為靜態場景、手臂保持固定姿態；全身預抓取移植與動態案例重跑仍待完成。')
    return f.save('04_模擬平台與驗證流程圖')


def results():
    f = Figure('05 / 成果','實驗成果：有障礙的全身協同預抓取','Gazebo・整合安全約束的單步 QP・μ = 0.03・固定截止時間排程・相同案例重複 3 次。')
    f.table(64,224,[700,480,612],['驗證指標','實測結果','說明'],[
        ('任務完成次數','3 / 3 次','同一起點、目標與靜態障礙配置'),
        ('完成時間','12.134 ± 0.023 s','三次測試平均 ± 標準差'),
        ('TCP 最終位置誤差','2.293 ± 0.008 mm','預抓取位置追蹤'),
        ('TCP 最終姿態誤差','0.055 ± 0.001°','工具朝向追蹤'),
        ('模型量測最小安全間距','約 88.8 mm','依模型距離指標量測'),
        ('QP 求解時間 p95','6.1 ms','求解耗時，非整條控制鏈延遲'),
        ('命令發布間隔：中位 / p95','50.0 / 52.6 ms','標稱控制頻率 20 Hz'),
    ],82,[29,31,27])
    f.text(86,903,'資料：baseline_B_frozen_20260909.md「排程修正版」；下游安全濾波器保留於閉迴路。',23,MUTED)
    f.footer('主要成果：建立可重複完成本案例的全身預抓取控制基線。','結果限於單一已知靜態場景；尚未完成實際夾持、跨場景成功率與實機驗證，亦非硬即時保證。')
    return f.save('05_全身預抓取成果表')


def isaac_table():
    f = Figure('補充 / 成果','Isaac Sim：時間修正後的三方向導航測試','固定障礙物・相同起點・三個 3 m 目標方向・手臂保持固定姿態。')
    f.table(64,256,[460,490,390,452],['目標方向','到達時間','終點位置誤差','到達判定'],[
        ('前方 0°','12.69 s','0.292 m','通過'),
        ('左側 90°','13.55 s','0.292 m','通過'),
        ('後方 180°','14.90 s','0.299 m','通過'),
    ],111,[32,36,34,32])
    f.box(64,707,866,177,'統一量測定義',['到達時間：自第一個 /plan 起算','判準：真值位置距目標 ≤ 0.30 m'],BLUE,size=29)
    f.box(962,707,894,177,'驗證範圍',['確認時間修正後的朝向導航行為','各方向僅一趟；未量跨場景成功率'],TEAL,size=29)
    f.footer('補充成果：三個方向皆符合到達判準，作為後續導航測試的介面基礎。','本表不屬動態避障或全身預抓取實驗；舊時間基準結果不可混入比較。')
    return f.save('06_Isaac三方向導航成果表')


def package(paths):
    # Small contact sheet for selection; the six originals remain at 4K.
    thumb_w, thumb_h = 800, 450
    sheet = Image.new('RGB',(1664,1565),'#EAF0F5')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',23)
    for i,p in enumerate(paths):
        x,y=24+(i%2)*816,24+(i//2)*514
        with Image.open(p.with_suffix('.png')) as im:
            assert im.size == (3840,2160)
            sheet.paste(im.convert('RGB').resize((thumb_w,thumb_h),Image.Resampling.LANCZOS),(x,y))
        draw.text((x+8,y+461),p.name.replace('_','  '),font=font,fill=NAVY)
    sheet.save(OUT/'00_全部圖表預覽.png')
    readme='''# 第五次進度報告圖表

每張原圖為 3840 × 2160 PNG（16:9、白底），可直接插入簡報。
同名 SVG 為可編輯向量原稿，PDF 為單頁向量輸出。SVG 使用 Noto Sans CJK TC；
若簡報電腦沒有此字型，建議使用 PNG，或使用已嵌入字型子集的 PDF。

01–04 對應四頁本次進度；05 對應一頁主要成果；06 為選用補充表。
00 為縮圖總覽，正式簡報請使用各張原圖。

## 圖片範圍

- 01 機器人為結構示意，不是 CAD 截圖，不依真實幾何比例。
- 02 連桿為幾何示意；圖示解釋距離與速度方向，不代表完整安全性證明。
- 03 顯示原架構與整合 QP 基線；完整到達同時涉及構形權重調整，下游濾波器仍保留。
- 04 顯示兩個模擬平台的工作分工，不代表已完成全身預抓取跨平台比較。
- 05 採 Gazebo 排程修正版三次重複測試；3/3 不是跨場景成功率。
- 06 僅採 Isaac 時鐘修正後的靜態三方向測試，不混用舊批次數據。

## 資料依據（相對專案根目錄）

- 第二階段進度紀錄.md：模型、運動學、自體濾波與安全層進度。
- evaluation/results/model_height_and_staleness_20260908.md：模型高度修正。
- evaluation/results/obstacle_wholebody_20260909.md：停滯與整合 QP 比較。
- evaluation/results/baseline_B_frozen_20260909.md：排程修正版成果表。
- evaluation/results/isaac_time_basis_fix_20260910.md：時間基準修正。
- evaluation/results/heading_probe_bigarena_20260910.md：修正後重跑數據。
- evaluation/results/isaac_base_camera_20260910.md：底盤相機介面。

產生程式：tools/build_fifth_progress_visuals.py
整理日期：2026-09-10。圖表依既有紀錄製作，本次未重跑模擬。
'''
    (OUT/'使用說明與數據來源.md').write_text(readme)
    archive=OUT.parent/'第五次進度報告_圖表素材.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file(): z.write(p,arcname='本次進度圖表/'+p.name)
        z.write(Path(__file__),arcname='產生程式/build_fifth_progress_visuals.py')
    print(f'Created {len(paths)} figures, each PNG / SVG / PDF; preview and ZIP: {archive}')


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    package([model(),safety(),control(),platforms(),results(),isaac_table()])
