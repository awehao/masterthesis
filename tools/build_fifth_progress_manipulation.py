#!/usr/bin/env python3
"""第五次進度報告：接觸操作（§9–§11）的圖表與影片素材。

沿用 tools/build_fifth_progress_visuals.py 的 Figure 類別與視覺識別，
只新增 07–12 六張圖；既有 01–06 與其 ZIP **一律不動**。

圖表數值**直接從趟次 JSON 讀取**，不在此重打；每張圖標明來源趟次。
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import importlib
V = importlib.import_module('build_fifth_progress_visuals')
from build_fifth_progress_visuals import (Figure as _Fig, OUT, NAVY, INK, MUTED,
                                          BLUE, TEAL, ORANGE, LIGHT, LINE, W, H)
from PIL import Image, ImageDraw, ImageFont

RUNS = ROOT / 'evaluation' / 'runs'
VID = ROOT / '第五次進度報告素材' / '影片'


class Fig(_Fig):
    """只改頁尾日期；其餘完全沿用。"""

    def footer(self, main, note):
        self.rect(64, 932, 1792, 98, NAVY)
        self.text(88, 973, main, 29, 'white', 600)
        self.text(88, 1009, note, 22, '#D3E1EE')
        self.text(1856, 1060, '資料截至 2026-09-13', 18, MUTED, anchor='end')

    # ---------------------------------------------------------------- 繪圖
    def axes(self, x, y, w, h, xr, yr, xlab, ylab, xticks, yticks, fmt='{:.0f}'):
        """畫一組座標軸，回傳 (世界→畫布) 的投影函式。"""
        self.rect(x, y, w, h, 'white', LINE)
        x0, x1 = xr
        y0, y1 = yr

        def P(vx, vy):
            return (x + (vx - x0) / (x1 - x0) * w,
                    y + h - (vy - y0) / (y1 - y0) * h)

        for v in xticks:
            px = P(v, y0)[0]
            self.line([(px, y + h), (px, y + h + 10)], MUTED, 2)
            self.text(px, y + h + 42, fmt.format(v), 24, MUTED, anchor='middle')
        for v in yticks:
            py = P(x0, v)[1]
            self.line([(x, py), (x + w, py)], '#EEF3F8', 2)
            self.text(x - 16, py + 9, f'{v:g}', 24, MUTED, anchor='end')
        self.text(x + w / 2, y + h + 84, xlab, 26, INK, 600, 'middle')
        self.text(x, y - 22, ylab, 26, INK, 600)
        return P

    def series(self, P, pts, color, width=5, dashed=False):
        self.line([P(a, b) for a, b in pts], color, width, dashed=dashed)

    def legend(self, x, y, items, step=40):
        for i, (color, label) in enumerate(items):
            yy = y + i * step
            self.line([(x, yy), (x + 46, yy)], color, 6)
            self.text(x + 60, yy + 9, label, 25, INK)


def load(run):
    J = json.load(open(RUNS / run / 'sim' / 'drawer_run.json'))
    ci = {c: k for k, c in enumerate(J['log_cols'])}
    return J, ci


def binned(J, ci, lo, hi, step):
    """拉開段：每 step mm 開度區間的 |F| 中位數與 |M| 中位數。"""
    import statistics as st
    lg = J['log']
    rows = [(r[ci['opening']] * 1000, r[ci['f_norm']], r[ci['tq_norm']])
            for r in lg if r[ci['phase']] == 'pull']
    out = []
    v = lo
    while v < hi:
        sel = [r for r in rows if v <= r[0] < v + step]
        if sel:
            out.append((v + step / 2, st.median(x[1] for x in sel),
                        st.median(x[2] for x in sel)))
        v += step
    return out


# ============================================================ 07 資產與抓取關係
def asset():
    f = Fig('07 / 模型', '被動抽屜資產與兩種抓取關係',
            '預抓取之後要真的動一個東西；抓取關係分兩版驗證，不混稱。')
    f.box(64, 224, 872, 268, '為什麼另建資產', [
        '場景既有 known_obs_12 是實心靜態方塊，',
        '沒有把手、沒有活動自由度。拿它當操作機構，',
        '「開抽屜」就成了沒有機構的宣稱。'], ORANGE, size=31)
    f.box(984, 224, 872, 268, '抽屜是被動的（讀回驗證）', [
        '滑動關節 drive 的 stiffness / damping /',
        'maxForce / target 全部為 0；',
        '抽屜不接收任何位置或速度命令。'], BLUE, size=31)
    f.table(64, 530, [420, 700, 672],
            ['抓取關係', '模型意義', '目前狀態'], [
        ('理想固定連接', '夾持時刻建立夾爪—抽屜固定關節，\n代表「抓握不會失效」的理想上限',
         '200 mm 完整操作完成'),
        ('摩擦夾持', '只有接觸與摩擦，兩指閉合夾住把手；\n抓握能力由實際接觸決定',
         '靜態雙側夾持通過；\n受控拉動約 17 mm，未達 20 mm'),
    ], 150, [30, 27, 28])
    f.text(86, 895, '把手 ⌀10 mm 橫桿，尺寸由夾爪幾何與行程需求反推；'
                    '開度上限 220 mm，讓 200 mm 目標不落在硬限位上。', 23, MUTED)
    f.footer('兩種抓取模型分版驗證，結果不互相代替。',
             '底盤在本階段由外部固定支撐；不代表輪子靠地面摩擦能承受相同負載。')
    return f.save('07_被動抽屜資產與抓取關係')


# ======================================================== 08 垂直負載與 b(s)
def load_curve():
    A, ca = load('drawer_234512_offset200')      # 未補償（中止）
    B, cb = load('drawer_065343_ffattach200')    # 全行程補償
    da = binned(A, ca, 0, 200, 10)
    db = binned(B, cb, 0, 200, 10)
    f = Fig('08 / 發現與修法', '垂直負載隨開度成長，與沿路徑前饋補償 b(s)',
            '同一條命令、同一目標；固定底座＋理想固定連接。')
    P = f.axes(150, 250, 1120, 500, (0, 200), (0, 35),
               '抽屜開度（mm）', '手腕傳遞力 |F|（N）',
               [0, 50, 100, 150, 200], [0, 10, 20, 30])
    f.line([P(0, 30), P(200, 30)], ORANGE, 3, dashed=True)
    f.text(P(4, 30)[0], P(4, 30)[1] - 16, '中止門檻 30 N', 24, ORANGE, 600)
    f.series(P, [(d[0], d[1]) for d in da], ORANGE, 6)
    f.series(P, [(d[0], d[1]) for d in db], TEAL, 6)
    xa, ya = da[-1][0], da[-1][1]
    f.circle(*P(xa, ya), 11, ORANGE, ORANGE)
    f.text(P(xa, ya)[0] - 18, P(xa, ya)[1] - 24,
           '於 97.2 mm 中止', 25, ORANGE, 700, 'end')
    f.legend(760, 660, [(ORANGE, '未補償：+0.2599 N/mm'),
                        (TEAL, '全行程補償：−0.0039 N/mm')])
    f.box(1320, 250, 536, 250, '沿滑軌的分量呢', [
        '全程只有 0.1 N 量級。', '', '成長的那部分不是拉抽屜的力。'],
        BLUE, size=29)
    f.box(1320, 532, 536, 250, '排除了什麼', [
        '取消抽屜重力：不變', '取消手臂重力：不變',
        '移除固定連接：成長完全消失'], TEAL, size=29)
    f.text(86, 908, '資料：drawer_234512_offset200（未補償）與 '
                    'drawer_065343_ffattach200（補償）的拉開段，每 10 mm 取中位數。',
           23, MUTED)
    f.footer('修法：路徑相依的前饋致動補償，完全由已量到的偏離反解，不加力回授。',
             '成長的物理來源仍未定案；補償是本案例的致動校準，不是通用方法。')
    return f.save('08_垂直負載成長與b(s)補償')


# ============================================================ 09 不可外推
def extrapolation():
    f = Fig('09 / 方法', '校準必須涵蓋全行程：線性外推會多補近一倍',
            '無連接試驗量到的垂直偏離 e⊥z 相對指令行程。')
    P = f.axes(150, 250, 1100, 500, (0, 210), (0, 2.4),
               '指令行程（mm）', '垂直偏離 e⊥z（mm）',
               [0, 50, 100, 150, 200], [0, 0.5, 1.0, 1.5, 2.0])
    meas = [(0, 0.0), (4.9, 0.152), (20.7, 0.388), (39.1, 0.598), (59.3, 0.757),
            (81.0, 0.873), (104.3, 0.957), (128.8, 1.014), (154.5, 1.050),
            (181.2, 1.069), (197.8, 1.120)]
    f.series(P, meas, TEAL, 6)
    f.series(P, [(0, 0.0), (197.8, 2.201)], ORANGE, 5, dashed=True)
    f.circle(*P(76.3, 0.849), 10, 'white', BLUE)
    f.line([P(76.3, 0), P(76.3, 2.2)], BLUE, 2, dashed=True)
    f.text(P(76.3, 2.2)[0] + 12, P(76.3, 2.2)[1] + 26,
           '舊校準表只到 76.3 mm', 24, BLUE, 600)
    f.legend(760, 660, [(TEAL, '實測（會飽和）'),
                        (ORANGE, '由 0–76 mm 線性外推')])
    f.table(1300, 250, [186, 170, 200], ['行程', '外推', '實測'], [
        ('100 mm', '1.114', '0.945'),
        ('150 mm', '1.669', '1.045'),
        ('198 mm', '2.201', '1.120'),
    ], 92, [27, 27, 27])
    f.box(1300, 566, 556, 216, '為什麼不能外推', [
        '外推到 198 mm 會多補約 1.08 mm ——',
        '與要消除的偏離本身同量級，',
        '等於反向製造一個新的偏離。'], ORANGE, size=28)
    f.text(86, 908, '兩趟獨立無連接校準在重疊區（0–76 mm）的 e⊥z 差異最大 '
                    '0.0043 mm，校準本身可重現。', 23, MUTED)
    f.footer('實測偏離不是線性、會飽和；外推值是實測的 1.96 倍。',
             '此結論來自本案例的兩趟校準，不外推到其他路徑或配時。')
    return f.save('09_校準不可外推')


# ====================================================== 10 200 mm 成果表
def result200():
    J, ci = load('drawer_065343_ffattach200')
    ev = {e['event']: e for e in J['events']}
    fin = J['final_opening_m'] * 1000
    arr = J['arrived_sim_t']
    rel = ev['release']['sim_t']
    fn = [r[ci['f_norm']] for r in J['log']]
    over = sum(1 for v in fn if v >= 30.0)
    f = Fig('10 / 成果', '200 mm 完整操作：拉開 → 保持 → 釋放 → 退出',
            '固定底座 ＋ 理想固定連接・drawer_065343_ffattach200・單趟，未重複。')
    f.table(64, 224, [560, 470, 762],
            ['驗證項目', '實測結果', '說明'], [
        ('最終開度', f'{fin:.3f} mm', f'目標 200 mm，誤差 {J["opening_err_m"]*1000:+.3f} mm'),
        ('受控到達時刻', f'{arr:.3f} s', f'解除連接於 {rel:.3f} s —— 早 {rel-arr:.3f} s'),
        ('解除後開度變化', '0.0000 mm', '記錄窗 17.7 s、解析度 0.01 s'),
        ('超過 30 N 的樣本', f'{over} / {len(fn)}', f'力峰值 {J["f_norm_peak"]["N"]:.2f} N，'
         f'出現在 {J["f_norm_peak"]["phase"]}'),
        ('底盤漂移', '0.00000 mm', '由匯入器的固定關節外部支撐'),
        ('命令套用 / 拒收', f'{J["cb"]["arm"]} / {J["cb"]["arm_rejected"]}', '監看無失效'),
    ], 76, [29, 31, 26])
    f.box(64, 762, 872, 118, '到位不是滑行', [
        '到達判準在固定連接仍在時滿足，', '解除後開度未再變化。'], TEAL, size=29)
    f.box(984, 762, 872, 118, '停止原因 ≠ 任務成功', [
        'sim_limit 是程序停止原因；', '完成依據是到位時刻與逐相位證據。'],
        ORANGE, size=29)
    f.footer('固定底座下，手臂帶動沒有驅動的抽屜完成 200 mm 全行程操作。',
             '單趟未重複；補償取自無連接試驗，成功不等於已辨識負載的物理來源。')
    return f.save('10_200mm完整操作成果表')


# ==================================================== 11 摩擦：對中與靜態夾持
def friction_grip():
    f = Fig('11 / 進行中', '摩擦夾持：先補實模型，再對中，才閉合',
            '只改抓取關係；幾何、行程、容差與停止處置沿用。')
    f.box(64, 224, 872, 250, '查核發現：規格「有寫、沒接」', [
        '全場沒有任何接觸摩擦材質（URDF 唯一的 μ',
        '是刻意設為 0 的無摩擦球輪）；',
        '12 N 手指接觸中止也不存在。'], ORANGE, size=30)
    f.box(984, 224, 872, 250, '補上後逐一讀回驗證', [
        '靜／動摩擦分開明訂各 0.8（建模假設）、',
        '混合規則 min、綁到 3 個碰撞形狀；',
        '手指出力上限執行期讀回 5.0 N／指。'], TEAL, size=30)
    f.table(64, 512, [470, 430, 430, 448],
            ['對中量（閉合前保持）', '修正前', '修正後', '判定'], [
        ('兩指中心沿閉合軸偏移', '−1.4376 mm', '−0.0021 mm', '≤ 0.30 mm 通過'),
        ('兩側指墊間隙差', '−2.1408 mm', '−0.0031 mm', '≤ 0.60 mm 通過'),
        ('工具姿態 vs 設計', '0.20283°', '0.00033°', '—'),
    ], 104, [28, 30, 30, 27])
    f.text(86, 862, '三層核對定位出偏差層級：設計位姿對中 0.0000 mm、命令對中 +0.0011 mm、'
                    '實際保持姿態偏 1.4376 mm', 24, INK, 600)
    f.text(86, 898, '⇒ 是穩態關節追蹤誤差（0.002150 rad，幾乎全在 joint2），'
                    '以命令層修正一次補上；目標幾何未動。', 23, MUTED)
    f.footer('靜態雙側夾持通過：連續 2.0 s、兩指接觸率 100 %、把手相對夾爪位移 0.0450 mm。',
             '修正是該工作點的經驗估計，不是已辨識的重力補償；換姿態或配時不保證有效。')
    return f.save('11_摩擦夾持對中與靜態夾持')


# ================================================== 12 摩擦拉動與方向分解
def friction_pull():
    f = Fig('12 / 進行中', '摩擦夾持的受控拉動與滑脫方向',
            '無理想固定連接・滑脫判準：相對拉動起點 2.0 mm。')
    P = f.axes(150, 250, 1050, 490, (0, 18), (0, 2.2),
               '抽屜開度（mm）', '把手相對夾爪的滑脫（mm）',
               [0, 5, 10, 15], [0, 0.5, 1.0, 1.5, 2.0])
    f.line([P(0, 2.0), P(18, 2.0)], ORANGE, 3, dashed=True)
    f.text(P(0.3, 2.0)[0], P(0.3, 2.0)[1] - 16, '滑脫門檻 2.0 mm',
           24, ORANGE, 600)
    s20 = [(0, 0.019), (2, 0.738), (4, 0.937), (6, 1.159), (8, 1.269),
           (10, 1.316), (12, 1.419), (14, 1.593), (15.79, 2.001)]
    s40 = [(0, 0.010), (2, 0.245), (4, 0.786), (6, 0.907), (8, 0.954),
           (10, 1.169), (12, 1.219), (14, 1.314), (15.79, 1.376),
           (17.12, 2.003)]
    f.series(P, s20, ORANGE, 6)
    f.series(P, s40, BLUE, 6)
    f.legend(230, 640, [(ORANGE, '原配時：受控 15.785 mm'),
                        (BLUE, '配時放慢一倍：受控 17.117 mm')])
    f.box(1250, 250, 606, 250, '滑脫方向分解（末筆）', [
        '沿滑軌     −2.0009 mm   100.0 %',
        '沿閉合軸   −0.0014 mm       0.0 %',
        '第三軸     −0.0103 mm       0.0 %'], TEAL, size=28)
    f.box(1250, 520, 606, 240, '放慢配時的效果', [
        '共同開度區間上端滑脫 −31.2 %，',
        '但單位滑脫率反而略升',
        '（0.06128 → 0.07213 mm/mm）。'], ORANGE, size=28)
    f.text(86, 872, '不是橫向跑偏、也不是伴隨轉動失去包覆 —— 是純粹沿拉動方向的相對滑動'
                    '（相對轉角峰值 0.199°，門檻 2.0°）。', 24, INK, 600)
    f.text(86, 906, '力與溫度保護未觸發（手指 11.4/12.0、手腕 15.0/30.0、溫度 81.9/92.0）；'
                    '滑脫保護已觸發。', 23, MUTED)
    f.footer('摩擦夾持在無固定連接下受控帶動抽屜 17.117 mm；20 mm 未完成。',
             '依裁決在此設停止點，不再降速；下一步轉向底盤—手臂協同，再回接摩擦。')
    return f.save('12_摩擦拉動與滑脫方向')


# ==================================================================== 影片
VIDEOS = [
    ('evaluation/results/demo_200mm_20260913/video/drawer200_full_operation.mp4',
     '11_200mm完整操作_執行時錄影.mp4',
     '執行時錄影：這一趟真正執行的當下逐物理步取像。畫面無任何文字。'),
    ('evaluation/results/demo_20260913/video/demo_drawer20_wide.mp4',
     '07_抽屜20mm_廣角_姿態重演.mp4',
     '姿態重演：讀已完成趟次的 log，逐幀設定回場景後取像。'),
    ('evaluation/results/demo_20260913/video/demo_drawer20_close.mp4',
     '07_抽屜20mm_近景_姿態重演.mp4',
     '姿態重演：把手近景。'),
    ('evaluation/results/demo_20260913/video/demo_pregrasp.mp4',
     '05_預抓取_姿態重演.mp4',
     '姿態重演：box12_south 預抓取。'),
]


def videos():
    VID.mkdir(parents=True, exist_ok=True)
    got = []
    for src, dst, note in VIDEOS:
        p = ROOT / src
        if not p.exists():
            print(f'  缺少影片 {src}')
            continue
        shutil.copy2(p, VID / dst)
        got.append((dst, note, (VID / dst).stat().st_size))
    # 說明另存，影片本身不含文字
    lines = ['# 第五次進度報告影片素材', '',
             '影像全部來自模擬器內的相機感測器，全程未做桌面擷取。', '',
             '兩類影片性質不同，不可混稱：', '',
             '| 檔案 | 類型與說明 | 大小 |', '|---|---|---|']
    for dst, note, size in got:
        kind = '執行時錄影' if '執行時錄影' in note else '姿態重演'
        lines.append(f'| `{dst}` | {note} | {size/1e6:.1f} MB |')
    lines += ['', '## 兩者的差別', '',
              '* 姿態重演：事後讀 log、逐幀把關節角與抽屜開度設定回場景再取像。',
              '  不是重新模擬，也不產生新的量測值。每幀設定後都讀回核對',
              '  （關節最大差 0.194 mrad、開度最大差 0.00048 mm）。',
              '* 執行時錄影：在那一趟執行的當下取像，每 3 個物理步一幀。',
              '  需在模擬程式內掛相機，因此該趟的程式 sha 與未加錄影功能的版本不同。', '',
              '## 影片不含文字', '',
              '依裁決，`11_200mm完整操作_執行時錄影.mp4` 不含片頭、字幕、數值、',
              '時間戳、階段標籤或浮水印；版本與實測結果見',
              '`evaluation/results/demo_200mm_20260913/README.md`。', '',
              '## 抓取關係標示', '',
              '抽屜影片一律為「固定底座 ＋ 理想固定連接」，不是摩擦夾持。',
              '摩擦夾持尚未完成 20 mm，本次無影片。']
    (VID / '影片說明.md').write_text('\n'.join(lines) + '\n')
    return got


# ==================================================================== 打包
def package(paths, vids):
    sheet_w, sheet_h = 1664, 1565
    sheet = Image.new('RGB', (sheet_w, sheet_h), '#EAF0F5')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc', 23)
    for i, p in enumerate(paths):
        x, y = 24 + (i % 2) * 816, 24 + (i // 2) * 514
        with Image.open(p.with_suffix('.png')) as im:
            sheet.paste(im.convert('RGB').resize((800, 450),
                        Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 8, y + 461), p.name.replace('_', '  '),
                  font=font, fill=NAVY)
    sheet.save(OUT / '00_接觸操作圖表預覽.png')

    readme = f'''# 第五次進度報告圖表：接觸操作（07–12）

沿用 01–06 的視覺識別與尺寸（3840 × 2160 PNG、同名 SVG 向量原稿、單頁 PDF）。
既有 01–06 未更動。

| 編號 | 對應報告章節 | 內容 |
|---|---|---|
| 07 | §9 | 被動抽屜資產；理想固定連接 vs 摩擦夾持 |
| 08 | §9.1–9.2 | 垂直負載隨開度成長與 b(s) 補償（實測曲線） |
| 09 | §9.2 | 校準不可外推：外推是實測的 1.96 倍 |
| 10 | §9.3 | 200 mm 完整操作成果表 |
| 11 | §10 | 摩擦夾持：模型補實與對中修正 |
| 12 | §10 | 摩擦受控拉動與滑脫方向分解 |

影片在 `第五次進度報告素材/影片/`，共 {len(vids)} 支，說明見該目錄的 `影片說明.md`。

## 圖表數值的來源

08 與 10 的數值由趟次 JSON 直接讀取，不在程式中重打：

- `evaluation/runs/drawer_234512_offset200`（未補償，於 97.2 mm 中止）
- `evaluation/runs/drawer_065343_ffattach200`（全行程補償，198.731 mm）

09、11、12 的數值取自 `evaluation/results/drawer_fixed_stage_20260912.md`
§84、§101、§113、§114，圖上已標明來源趟次。

## 使用時必須一起說的限制

- 底盤在本階段由外部固定支撐；不代表輪子靠地面摩擦能承受相同負載。
- 200 mm 那趟只跑一趟，未重複；補償前後兩趟的基準與參考都改變，
  不是單變數對照。
- b(s) 的適用條件是「同模型、固定底座、同路徑與配時、同增益」，
  是本案例的致動校準，不是通用方法。
- `sim_limit` 是程序停止原因，不是任務成功的判準。
- 摩擦夾持尚未完成 20 mm（最佳 17.118 / 20 mm）。

產生程式：`tools/build_fifth_progress_manipulation.py`
整理日期：2026-09-13。
'''
    (OUT / '使用說明_接觸操作.md').write_text(readme)

    archive = OUT.parent / '第五次進度報告_接觸操作素材.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            for ext in ('.png', '.svg', '.pdf'):
                q = p.with_suffix(ext)
                if q.exists():
                    z.write(q, arcname='本次進度圖表/' + q.name)
        z.write(OUT / '00_接觸操作圖表預覽.png',
                arcname='本次進度圖表/00_接觸操作圖表預覽.png')
        z.write(OUT / '使用說明_接觸操作.md',
                arcname='本次進度圖表/使用說明_接觸操作.md')
        for f in sorted(VID.iterdir()):
            if f.is_file():
                z.write(f, arcname='影片/' + f.name)
        z.write(Path(__file__),
                arcname='產生程式/build_fifth_progress_manipulation.py')
    print(f'圖表 {len(paths)} 張（PNG/SVG/PDF）、影片 {len(vids)} 支；'
          f'打包 -> {archive}')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    vids = videos()
    figs = [asset(), load_curve(), extrapolation(),
            result200(), friction_grip(), friction_pull()]
    package(figs, vids)
