#!/usr/bin/env python3
"""Build the fourth progress report in the visual rhythm of report three."""

import os
import time
import subprocess
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "第四次進度報告簡報.pptx"
PDF = ROOT / "第四次進度報告簡報.pdf"

W, H = 33867, 19050  # 16:9, 13.333 x 7.5 in in 1/100 mm
NAVY = 0x17365D
BLUE = 0x2F75B5
LIGHT_BLUE = 0xDDEBF7
PALE_BLUE = 0xEEF5FB
ORANGE = 0xED7D31
RED = 0xC00000
GREEN = 0x548235
DARK = 0x222222
GRAY = 0x666666
LIGHT_GRAY = 0xF2F2F2
WHITE = 0xFFFFFF


def pv(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


def connect_office():
    pipe_name = f"codex_progress_{os.getpid()}"
    subprocess.Popen(
        ["soffice", "--headless", f"--accept=pipe,name={pipe_name};urp;StarOffice.ComponentContext", "--norestore", "--nodefault", "--nofirststartwizard", f"-env:UserInstallation=file:///tmp/{pipe_name}_profile"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    for _ in range(40):
        try:
            ctx = resolver.resolve(f"uno:pipe,name={pipe_name};urp;StarOffice.ComponentContext")
            smgr = ctx.ServiceManager
            desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
            return desktop
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Unable to connect to LibreOffice")


def add_shape(page, doc, kind, x, y, w, h, fill=WHITE, line=WHITE, radius=False):
    service = "com.sun.star.drawing.RectangleShape" if kind == "rect" else kind
    s = doc.createInstance(service)
    s.Position = uno.createUnoStruct("com.sun.star.awt.Point", x, y)
    s.Size = uno.createUnoStruct("com.sun.star.awt.Size", w, h)
    if kind == "rect":
        s.FillColor = fill
        s.LineColor = line
        s.LineWidth = 20
        if radius:
            try:
                s.CornerRadius = 220
            except Exception:
                pass
    page.add(s)
    return s


def add_text(page, doc, text, x, y, w, h, size=18, color=DARK, bold=False,
             align=0, valign=0, fill=None, line=None, margin=180):
    s = doc.createInstance("com.sun.star.drawing.TextShape")
    s.Position = uno.createUnoStruct("com.sun.star.awt.Point", x, y)
    s.Size = uno.createUnoStruct("com.sun.star.awt.Size", w, h)
    s.String = text
    s.CharFontName = "Noto Sans CJK TC"
    s.CharHeight = float(size)
    s.CharColor = color
    s.CharWeight = 150.0 if bold else 100.0
    s.ParaAdjust = align
    s.TextVerticalAdjust = valign
    s.TextLeftDistance = margin
    s.TextRightDistance = margin
    s.TextUpperDistance = 100
    s.TextLowerDistance = 100
    if fill is None:
        s.FillStyle = 0
    else:
        s.FillStyle = 1
        s.FillColor = fill
    if line is None:
        s.LineStyle = 0
    else:
        s.LineStyle = 1
        s.LineColor = line
    page.add(s)
    return s


def add_line(page, doc, x1, y1, x2, y2, color=BLUE, width=30):
    s = doc.createInstance("com.sun.star.drawing.LineShape")
    s.Position = uno.createUnoStruct("com.sun.star.awt.Point", x1, y1)
    s.Size = uno.createUnoStruct("com.sun.star.awt.Size", x2 - x1, y2 - y1)
    s.LineColor = color
    s.LineWidth = width
    page.add(s)
    return s


def add_image(page, doc, path, x, y, w, h):
    s = doc.createInstance("com.sun.star.drawing.GraphicObjectShape")
    s.Position = uno.createUnoStruct("com.sun.star.awt.Point", x, y)
    s.Size = uno.createUnoStruct("com.sun.star.awt.Size", w, h)
    s.GraphicURL = uno.systemPathToFileUrl(str(path.resolve()))
    page.add(s)
    return s


def base_slide(doc, title, num):
    pages = doc.getDrawPages()
    page = pages.insertNewByIndex(pages.getCount())
    page.Width, page.Height = W, H
    add_text(page, doc, title, 1200, 650, 31000, 1300, 27, NAVY, True)
    add_line(page, doc, 1200, 2050, 31000, 2050, LIGHT_BLUE, 35)
    add_text(page, doc, "2026/08", 850, 18100, 3500, 500, 10, GRAY)
    add_text(page, doc, str(num), 31700, 18100, 700, 500, 10, GRAY, align=1)
    return page


def problem_solution(page, doc, problem, solution, result, result_color=GREEN):
    cols = [(1100, 8550), (12650, 8550), (24200, 8550)]
    heads = [("原有問題", RED), ("解決方法", BLUE), ("驗證結果", result_color)]
    bodies = [problem, solution, result]
    for (x, w), (head, color), body in zip(cols, heads, bodies):
        add_text(page, doc, head, x, 2650, w, 850, 18, WHITE, True, align=1, valign=2, fill=color, line=color)
        add_text(page, doc, body, x, 3600, w, 11600, 16, DARK, False, fill=0xFAFAFA, line=0xD9E2F3, margin=350)


def add_table(page, doc, x, y, widths, rows, row_h=950, header=True, fs=13):
    yy = y
    for r, row in enumerate(rows):
        xx = x
        for c, value in enumerate(row):
            fill = NAVY if r == 0 and header else (PALE_BLUE if r % 2 == 1 else WHITE)
            color = WHITE if r == 0 and header else DARK
            add_text(page, doc, str(value), xx, yy, widths[c], row_h, fs, color,
                     bold=(r == 0 and header), align=1, valign=2, fill=fill, line=0xC9D5E5, margin=90)
            xx += widths[c]
        yy += row_h


def build():
    desktop = connect_office()
    doc = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, ())
    pages = doc.getDrawPages()
    while pages.getCount():
        pages.remove(pages.getByIndex(0))

    # 1 title
    p = pages.insertNewByIndex(0); p.Width, p.Height = W, H
    add_text(p, doc, "Predictive Navigation and Constrained Manipulation of\nMobile Manipulators in Dynamic Environments", 1800, 2800, 30200, 2600, 24, NAVY, True, align=1, valign=2)
    add_text(p, doc, "動態環境下移動機械臂預測導航與受約束操作", 2500, 5700, 28800, 1300, 27, NAVY, True, align=1)
    add_line(p, doc, 7000, 7350, 26800, 7350, ORANGE, 55)
    add_text(p, doc, "第四次進度報告", 9500, 8000, 14800, 1100, 23, ORANGE, True, align=1)
    add_text(p, doc, "報告者：陳文顥\n指導教授：陳榮順 教授", 10600, 10600, 12600, 1700, 16, DARK, align=1)
    add_text(p, doc, "2026/08", 1000, 18100, 3000, 500, 10, GRAY)
    add_text(p, doc, "1", 31700, 18100, 700, 500, 10, GRAY, align=1)

    # 2 outline
    p = base_slide(doc, "報告大綱", 2)
    for i, (t, sub) in enumerate([
        ("進度回顧", "第三次報告完成了什麼"),
        ("本次進度", "問題、修正與實驗證據"),
        ("未來工作", "raw-scan safety shield 驗證"),
    ]):
        y = 3600 + i * 3800
        add_text(p, doc, str(i+1), 4000, y, 1500, 1500, 28, WHITE, True, align=1, valign=2, fill=BLUE, line=BLUE)
        add_text(p, doc, t, 6100, y, 6500, 850, 21, NAVY, True)
        add_text(p, doc, sub, 6100, y+900, 19000, 650, 15, GRAY)

    # 3 goal
    p = base_slide(doc, "研究目標", 3)
    add_text(p, doc, "第一階段：導航避障", 1600, 2900, 9500, 900, 20, BLUE, True)
    add_text(p, doc, "動態未知環境中安全抵達定點\nSE(2) GMPC＋動態預測式 CBF\nLiDAR 障礙追蹤＋穩健定位", 1600, 4000, 9800, 3600, 17, DARK)
    add_text(p, doc, "第二階段：全身協同操作", 21900, 2900, 9800, 900, 20, ORANGE, True)
    add_text(p, doc, "底盤 SE(2)＋手臂 SE(3) 同時求解\n開門、開抽屜的鉸接物件約束\n整合避障、關節極限與操作度", 21900, 4000, 9800, 3600, 17, DARK)
    stages = ["前往書房", "避開未知\n動靜態障礙", "抵達抽屜", "開啟並夾取", "安全返回"]
    for i, s in enumerate(stages):
        x = 1300 + i * 6450
        add_text(p, doc, s, x, 10500, 4800, 2200, 16, WHITE, True, align=1, valign=2, fill=(BLUE if i < 3 else ORANGE), line=WHITE)
        if i < 4:
            add_text(p, doc, "→", x+4950, 11100, 1400, 900, 25, NAVY, True, align=1)
    add_text(p, doc, "本期仍聚焦第一階段：讓安全導航的結論能泛化到未參與調參的新場景", 2600, 14800, 28600, 900, 17, NAVY, True, align=1, fill=PALE_BLUE, line=LIGHT_BLUE)

    # 4 architecture
    p = base_slide(doc, "導航系統架構", 4)
    layers = [
        ("Layer 6｜最後安全層", "raw-scan safety shield：不依賴分類，只限制朝近距離回波的速度", ORANGE),
        ("Layer 5｜控制", "SE(2) GMPC：全向底盤軌跡追蹤", BLUE),
        ("Layer 4｜安全", "全時域預測式 CBF：利用障礙速度建立預測約束", BLUE),
        ("Layer 3｜感知", "LiDAR cluster → track → 動靜分類 → 表面點", BLUE),
        ("Layer 2｜規劃", "Nav2 全域規劃＋costmap：已知與未知靜態繞行", BLUE),
        ("Layer 1｜定位", "AMCL beam-skip＋EKF：動態環境抗漂移定位", BLUE),
    ]
    for i, (head, body, color) in enumerate(layers):
        y = 2600 + i * 2350
        add_text(p, doc, head, 3300, y, 7200, 1150, 17, WHITE, True, align=1, valign=2, fill=color, line=color)
        add_text(p, doc, body, 10500, y, 19800, 1150, 16, DARK, valign=2, fill=(0xFFF2E8 if color == ORANGE else PALE_BLUE), line=color)

    # 5 recap
    p = base_slide(doc, "進度回顧：第三次報告的結束狀態", 5)
    problem_solution(p, doc,
        "第二次報告仍使用 Gazebo 真值障礙物，無法代表真實感知鏈；點質點模型也無法反映實際車體與 LiDAR 自遮。",
        "導入 omni_bot 實機尺寸與 2D LiDAR；建立 scan obstacle tracker、AMCL beam-skip＋EKF、map-based static CBF。",
        "舊場景 N=40：\n到達 40/40\n動態碰撞 0\n總碰撞 1/40\n平均求解 2.2 ms")
    add_text(p, doc, "第三次的 98% 是單一舊場景結果；第四次工作的核心是檢驗它能否泛化。", 3100, 15700, 27600, 900, 18, RED, True, align=1, fill=0xFCE4D6, line=0xF4B183)

    # 6 generalization
    p = base_slide(doc, "本次進度：場景升級與泛化測試", 6)
    img = ROOT / "evaluation/results/figs/bigarena.png"
    if img.exists():
        add_image(p, doc, img, 1250, 3000, 12500, 10500)
    problem_solution(p, doc,
        "舊場景為開放房間、固定起訖點與 4 個移動體，路線單一，容易掩蓋門洞、交會時序與未知物體的失效。",
        "新增 bigarena：8 個隔間、交錯門洞、18 個已知箱體、7 個未知靜物、10 個移動體，起訖點隨機取樣。",
        "相同架構在新場景曾出現 20–41% 接觸。證明舊場景的高成功率不能直接代表泛化能力。")
    # Hide first card area behind image; retain three-card rhythm on right using compact overlay
    add_shape(p, doc, "rect", 14200, 3000, 17700, 11700, WHITE, WHITE)
    add_text(p, doc, "原有問題", 14500, 3200, 5200, 700, 17, WHITE, True, align=1, valign=2, fill=RED, line=RED)
    add_text(p, doc, "舊場景開放、固定路線，難以暴露門洞與多物體交會失效。", 14500, 4000, 5200, 3200, 15, DARK, fill=LIGHT_GRAY, line=0xDDDDDD)
    add_text(p, doc, "解決方法", 20300, 3200, 5200, 700, 17, WHITE, True, align=1, valign=2, fill=BLUE, line=BLUE)
    add_text(p, doc, "建立 bigarena 與隨機起訖；跨配置固定使用相同路線。", 20300, 4000, 5200, 3200, 15, DARK, fill=PALE_BLUE, line=LIGHT_BLUE)
    add_text(p, doc, "驗證結果", 26100, 3200, 5200, 700, 17, WHITE, True, align=1, valign=2, fill=GREEN, line=GREEN)
    add_text(p, doc, "新場景接觸率升高，成功揭露第三次報告沒有看到的結構性問題。", 26100, 4000, 5200, 3200, 15, DARK, fill=0xE2F0D9, line=0xA9D18E)
    add_text(p, doc, "bigarena：更高場景多樣性，而不是刻意把障礙物放在機器人路徑上", 14800, 9200, 16000, 1600, 17, NAVY, True, align=1, valign=2, fill=PALE_BLUE, line=LIGHT_BLUE)

    # 7 hardware
    p = base_slide(doc, "本次進度：硬體限制修正", 7)
    problem_solution(p, doc,
        "控制器設定與實際輪系不一致：vx 超出硬體 26%；加速度僅使用約 1/8；velocity smoother 又夾住後退命令。",
        "由輪半徑、幾何尺寸與輪速上限重推底盤速度／加速度，並統一 GMPC、baseline 與 smoother 的限制。",
        "輪速多面體缺失、錯誤硬體上限與下游覆寫皆已修復。舊批次不可再作跨方法效能比較。")
    add_table(p, doc, 4500, 13200, [6200, 6200, 6200], [
        ["項目", "舊設定", "硬體推導"],
        ["vx / vy", "最高 0.35 m/s", "0.2775 m/s"],
        ["ax / ay", "0.8 / 0.6 m/s²", "6.25 / 6.25 m/s²"],
        ["yaw rate", "0.80 rad/s", "1.1327 rad/s"],
    ], row_h=760, fs=12)

    # 8 measurement
    p = base_slide(doc, "本次進度：重建量測方法", 8)
    problem_solution(p, doc,
        "跨場景結果混入同一 CSV、到達後停車仍計碰撞、模擬碰撞體與分析半徑不同、只確認定位有發布卻不量誤差。",
        "每批獨立結果；只統計任務到達前；碰撞體統一為 r=0.30 m 圓盤；加入 AMCL/EKF 對真值誤差與靜／動接觸分離。",
        "−0.300 m ×49 的深穿透全為跨場景污染；bigarena 199 趟實際靜態負間距僅 18 趟，最深 −0.050 m。")
    add_text(p, doc, "量測工具本身也是實驗系統的一部分：錯誤統計會把不存在的問題變成主要問題。", 2500, 15800, 28800, 800, 17, NAVY, True, align=1, fill=PALE_BLUE, line=LIGHT_BLUE)

    # 9 overnight
    p = base_slide(doc, "本次進度：硬／軟 CBF 與定位消融", 9)
    add_table(p, doc, 1200, 3000, [6100, 2700, 3300, 3300, 3300, 3300], [
        ["設定", "n", "到達", "接觸", "靜態", "動態"],
        ["硬靜態 CBF", "38", "38", "5", "2", "3"],
        ["軟 slack", "38", "37", "6", "2", "4"],
        ["硬靜態重跑", "39", "39", "4", "0", "4"],
        ["軟 slack＋EKF 位姿", "35", "35", "6", "0", "6"],
    ], row_h=1050, fs=14)
    problem_solution(p, doc,
        "曾懷疑碰撞來自 slack 定價、定位失步或硬靜態約束不足。",
        "同一路線做硬／軟 CBF、重複實驗與位姿來源消融；逐週期核對 QP 約束。",
        "沒有任何一項顯著改變接觸率。McNemar p=1.0；同設定重跑 5 vs 4，顯示小差異低於雜訊地板。")
    add_shape(p, doc, "rect", 800, 8500, 32200, 7600, WHITE, WHITE)
    add_text(p, doc, "結論", 3000, 9200, 4500, 800, 18, WHITE, True, align=1, valign=2, fill=NAVY, line=NAVY)
    add_text(p, doc, "CBF 強弱與定位不是共同根因；必須回到每次殘餘碰撞，檢查障礙物是否真的進入 CBF。", 7600, 9200, 23000, 800, 17, DARK, True, valign=2, fill=PALE_BLUE, line=LIGHT_BLUE)
    add_text(p, doc, "同路線同設定的間距差：中位 0.043 m、最大 0.479 m\n→ n≈40 時，碰撞數相差 1–3 次不足以構成機制證據。", 4200, 11200, 25200, 2600, 18, RED, True, align=1, valign=2, fill=0xFCE4D6, line=0xF4B183)

    # 10 collision decomposition
    p = base_slide(doc, "本次進度：殘餘碰撞拆解", 10)
    add_text(p, doc, "115 趟、15 次接觸", 1100, 2800, 8500, 1000, 22, NAVY, True)
    add_text(p, doc, "靜態 4", 1700, 4600, 7600, 1600, 24, WHITE, True, align=1, valign=2, fill=BLUE, line=BLUE)
    add_text(p, doc, "0.008、0.050、0.050、0.050 m\n稀少且極淺，p=0.12", 1700, 6400, 7600, 2200, 16, DARK, align=1, valign=2, fill=PALE_BLUE, line=LIGHT_BLUE)
    add_text(p, doc, "動態 11", 10500, 4600, 7600, 1600, 24, WHITE, True, align=1, valign=2, fill=ORANGE, line=ORANGE)
    add_text(p, doc, "6 次淺於 2 cm\n3 次深於 12 cm", 10500, 6400, 7600, 2200, 16, DARK, align=1, valign=2, fill=0xFFF2E8, line=0xF4B183)
    add_text(p, doc, "唯一穩定重現", 19300, 4600, 11700, 1600, 24, WHITE, True, align=1, valign=2, fill=RED, line=RED)
    add_text(p, doc, "seed27：四組皆碰撞\n唯一穿透趟間雜訊的個案", 19300, 6400, 11700, 2200, 16, DARK, align=1, valign=2, fill=0xFCE4D6, line=0xF4B183)
    add_text(p, doc, "策略：不再用總碰撞率猜原因，改用可重現 seed27 做逐週期因果追蹤。", 3000, 11200, 27800, 1200, 19, NAVY, True, align=1, valign=2, fill=PALE_BLUE, line=LIGHT_BLUE)

    # 11 root cause
    p = base_slide(doc, "本次進度：找到 seed27 的真正原因", 11)
    problem_solution(p, doc,
        "原先推測大型物體跨遮罩造成 cluster 碎裂、軌跡失聯與 coast 零發布；但 debug 顯示軌跡 100% 存在、age 中位 218、KF 速度 0.096→0.097 m/s。",
        "新增 track ID、age、速度、innovation 與發布狀態診斷，直接沿著 cluster→track→分類→CBF 的每一道 gate 查驗。",
        "真正卡在瞬時速度門檻：min_track_speed=0.10，而 dyn_obs_5 約 0.10；2秒淨位移門檻 0.05 明明已通過，卻仍被排除。")
    add_text(p, doc, "cluster → track → age≥3 → 瞬時速度≥0.10 → 淨位移≥0.05 → 發布給 CBF", 1900, 15500, 30000, 950, 18, WHITE, True, align=1, valign=2, fill=NAVY, line=NAVY)

    # 12 speed gate
    p = base_slide(doc, "本次進度：移除重複速度閘門", 12)
    problem_solution(p, doc,
        "瞬時 KF 速度容易在閾值附近抖動；任何一關未通過，障礙物就不是『約束較弱』，而是完全不在 CBF 集合中。",
        "先以最小消融將 min_track_speed 0.10→0.05，其餘 tracker 行為不動；長期由 2 秒淨位移負責排除靜物。",
        "同一 seed27 重播 10 次，先驗證 dyn_obs_5 的確認發布率是否由約 29% 明顯上升，再看接觸與到達。", result_color=ORANGE)
    add_text(p, doc, "機制指標優先：發布率、拒絕原因、ID switch、innovation、速度 SD\n結果指標其次：最小間距、碰撞、到達", 4000, 15100, 25800, 1500, 17, NAVY, True, align=1, valign=2, fill=PALE_BLUE, line=LIGHT_BLUE)

    # 13 shield architecture
    p = base_slide(doc, "本次進度：raw-scan safety shield", 13)
    problem_solution(p, doc,
        "逐一修 cluster、track、age 與速度 gate，仍無法保證下一個障礙不會在分類鏈的其他位置被完全丟棄。",
        "在 velocity smoother 後、底盤前加入獨立 safety shield；直接讀 raw /scan，只要近距離有效回波存在，就限制朝它接近的速度。",
        "分類鏈負責預測與高效率繞行；raw shield 負責分類失敗時仍不允許立即接觸。設計與單元測試完成，待場景驗證。", result_color=ORANGE)
    add_text(p, doc, "/scan → 有效性／自身回波／相鄰一致性 → 固定 6 輪投影 → 最終速度命令", 2100, 15300, 29600, 900, 17, WHITE, True, align=1, valign=2, fill=ORANGE, line=ORANGE)

    # 14 shield math
    p = base_slide(doc, "安全盾約束與即時性設計", 14)
    add_text(p, doc, "nᵢᵀ(v + ωJrᵢ) ≤ α(dᵢ − dstop,ᵢ)", 3200, 3000, 27500, 1300, 26, NAVY, True, align=1, valign=2, fill=PALE_BLUE, line=LIGHT_BLUE)
    add_text(p, doc, "dstop = d₀ + vτ + v²/(2abrake) + εscan + εfootprint", 4500, 4700, 25000, 1050, 20, DARK, True, align=1)
    cards = [
        ("幾何", "n 指向障礙物；保留旋轉項。圓盤自轉項為零，但非圓形輪廓時不可省略。"),
        ("即時", "固定次數逐次投影，不使用一般 QP；執行時間有界。最後檢查最大約束殘差。"),
        ("感知", "直接讀 raw /scan；緊急距離內單點即可觸發，不需要 cluster、track 或 age。"),
        ("逾時", "scan >0.5 s 時將平移速度範數限制為 0.05 m/s，仍允許爬行與轉向脫困。"),
    ]
    for i, (head, body) in enumerate(cards):
        x = 1600 + (i % 2) * 15800
        y = 7000 + (i // 2) * 4300
        add_text(p, doc, head, x, y, 3600, 900, 17, WHITE, True, align=1, valign=2, fill=(BLUE if i < 2 else ORANGE), line=WHITE)
        add_text(p, doc, body, x+3700, y, 11200, 2400, 15, DARK, valign=2, fill=LIGHT_GRAY, line=0xD9E2F3)

    # 15 validation
    p = base_slide(doc, "驗證計畫", 15)
    add_table(p, doc, 1200, 2800, [5600, 12300, 12800], [
        ["階段", "要回答的問題", "主要指標"],
        ["S1 速度閘門", "分類鏈是否真的漏掉慢速物體？", "dyn_obs_5 發布率、拒絕原因、ID switch"],
        ["Shield 單元／針對性測試", "最後命令是否滿足約束且不妨礙切向／遠離？", "max residual、觸發距離、修改量、scan age"],
        ["seed27 重播", "即使 tracker 不發布，raw shield 能否阻止深穿透？", "最小間距、到達、shield activation"],
        ["100 趟未知場景", "整體架構能否在未調參場景維持安全？", "到達率、靜／動碰撞、95% 區間"],
    ], row_h=1500, fs=13)
    add_text(p, doc, "0/100 代表『100 趟未觀察到碰撞』；零事件的 95% 上界約 3%，不宣稱絕對碰撞率為零。", 2300, 15700, 29200, 800, 16, RED, True, align=1, fill=0xFCE4D6, line=0xF4B183)

    # 16 current progress
    p = base_slide(doc, "目前進度成果", 16)
    add_text(p, doc, "完成", 2600, 3000, 5200, 1000, 21, WHITE, True, align=1, valign=2, fill=GREEN, line=GREEN)
    add_text(p, doc, "✓ bigarena 泛化場景與隨機路線\n✓ 硬體速度、加速度與輪速耦合修正\n✓ 車體幾何、量測窗口與跨場景污染修正\n✓ 115 趟殘餘碰撞拆解\n✓ seed27 分類鏈根因定位\n✓ raw-scan shield 實作與診斷", 2300, 4300, 12600, 7600, 17, DARK, fill=0xE2F0D9, line=0xA9D18E)
    add_text(p, doc, "進行中", 18000, 3000, 5200, 1000, 21, WHITE, True, align=1, valign=2, fill=ORANGE, line=ORANGE)
    add_text(p, doc, "• 速度閘門 seed27 重播\n• Shield 正向／切向／門柱／慢速物體測試\n• seed27＋shield 因果驗證\n• 100 趟未知動態場景驗證", 17700, 4300, 13400, 7600, 17, DARK, fill=0xFFF2E8, line=0xF4B183)
    add_text(p, doc, "本期最大的進展：從『調參降低碰撞』轉為『以逐週期證據找出安全約束在哪裡消失』。", 2800, 14300, 28200, 1500, 19, NAVY, True, align=1, valign=2, fill=PALE_BLUE, line=LIGHT_BLUE)

    # 17 future
    p = base_slide(doc, "未來工作", 17)
    future = [
        ("1", "完成 safety shield 驗證", "證明分類失效時仍可由 raw scan 保持近距離安全"),
        ("2", "建立公平正式基準", "GMPC／MPPI／RPP 使用相同硬體限制、路線與量測窗口"),
        ("3", "擴充至全身協同", "將 SE(2) 底盤 GMPC 擴充至底盤＋手臂的受約束操作"),
    ]
    for i, (n, head, body) in enumerate(future):
        y = 3400 + i * 4200
        add_text(p, doc, n, 2700, y, 1800, 1800, 28, WHITE, True, align=1, valign=2, fill=(BLUE if i < 2 else ORANGE), line=WHITE)
        add_text(p, doc, head, 5200, y, 9500, 850, 20, NAVY, True)
        add_text(p, doc, body, 5200, y+1000, 24000, 900, 16, GRAY)

    # 18 thanks
    p = pages.insertNewByIndex(pages.getCount()); p.Width, p.Height = W, H
    add_text(p, doc, "Thank you!", 6000, 6900, 21800, 2300, 34, NAVY, True, align=1, valign=2)
    add_line(p, doc, 11200, 9600, 22600, 9600, ORANGE, 55)
    add_text(p, doc, "2026/08", 1000, 18100, 3000, 500, 10, GRAY)
    add_text(p, doc, "18", 31700, 18100, 700, 500, 10, GRAY, align=1)

    doc.storeAsURL(uno.systemPathToFileUrl(str(OUT)), (pv("FilterName", "Impress MS PowerPoint 2007 XML"), pv("Overwrite", True)))
    doc.storeToURL(uno.systemPathToFileUrl(str(PDF)), (pv("FilterName", "impress_pdf_Export"), pv("Overwrite", True)))
    doc.close(True)
    print(OUT)
    print(PDF)


if __name__ == "__main__":
    build()
