#!/usr/bin/env python3
"""Build an editable, evidence-scoped draft. No ROS, controller or simulator starts.

Run with /usr/bin/python3 (LibreOffice UNO). Reuses report-four drawing helpers.
Only presentation outputs and extracted replay stills are written here.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import uno
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ASSETS = OUT / "素材"
MAT = ROOT / "第七次進度報告素材"
spec = importlib.util.spec_from_file_location("previous_slide_helpers", ROOT / "tools/build_fourth_progress_slides.py")
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)

W, H = U.W, U.H
NAVY, BLUE, ORANGE = U.NAVY, U.BLUE, U.ORANGE
DARK, GRAY, WHITE = U.DARK, U.GRAY, U.WHITE
PALE, LINE, GREEN, RED = 0xF3F7FB, 0xDBE4EE, 0x287766, 0xAC3C36
SLIDES = []
DRAFT_VERSION = "v1"


def rect(p, x, y, w, h, fill=PALE, line=LINE):
    return U.add_shape(p, DOC, "rect", int(x), int(y), int(w), int(h), fill, line)


def txt(p, text, x, y, w, h, size=19, color=DARK, bold=False, align=0):
    s = U.add_text(p, DOC, text, int(x), int(y), int(w), int(h), size, color, bold, align=align, margin=120)
    # UNO initializes a drawing text object when it is added to the page.
    # Set content and character properties after insertion, not only beforehand.
    s.String = text
    s.CharFontName = "Noto Sans CJK TC"
    s.CharFontNameAsian = "Noto Sans CJK TC"
    s.CharHeight = float(size)
    s.CharHeightAsian = float(size)
    s.CharColor = color
    s.CharWeight = 150.0 if bold else 100.0
    s.CharWeightAsian = 150.0 if bold else 100.0
    s.ParaAdjust = align
    s.TextAutoGrowHeight = False
    s.TextAutoGrowWidth = False
    return s


def image_fit(p, file, x, y, w, h):
    with Image.open(file) as im:
        iw, ih = im.size
    k = min(w / iw, h / ih)
    return U.add_image(p, DOC, file, int(x + (w-iw*k)/2), int(y + (h-ih*k)/2), int(iw*k), int(ih*k))


def slide(title, section, notes, sources=()):
    pages = DOC.getDrawPages()
    p = pages.getByIndex(0) if not SLIDES else pages.insertNewByIndex(pages.getCount())
    while p.getCount():
        p.remove(p.getByIndex(0))
    p.Width, p.Height = W, H
    n = len(SLIDES)+1
    txt(p, section, 1250, 470, 28000, 500, 11, BLUE, True)
    title_h, divider_y = (2500, 4000) if "\n" in title else (1500, 2800)
    txt(p, title, 1250, 1140, 31300, title_h, 28, NAVY, True)
    U.add_line(p, DOC, 1250, divider_y, W-1250, divider_y, LINE, 30)
    txt(p, f"第七次進度報告｜工作稿 {DRAFT_VERSION}｜2026/10/28 整理・報告日期待填", 1250, 18000, 29000, 550, 9, GRAY)
    txt(p, str(n), W-2400, 18000, 1000, 550, 11, GRAY, align=2)
    SLIDES.append(dict(n=n, title=title, section=section, notes=notes, sources=list(sources)))
    print(f"Slide {n:02d}: {title.replace(chr(10), ' ')}", flush=True)
    np = p.getNotesPage()
    body = next((np.getByIndex(i) for i in range(np.getCount())
                 if np.getByIndex(i).getShapeType() == "com.sun.star.presentation.NotesShape"), None)
    if body is not None:
        body.String = notes + "\n\n來源：\n" + "\n".join(sources)
    return p


def banner(p, message, y=16200, color=NAVY, fill=PALE):
    rect(p, 1250, y, W-2500, 1150, fill, fill)
    txt(p, message, 1550, y+120, W-3100, 950, 16, color, True)


def card(p, heading, body, x, y, w, h=3600, color=BLUE):
    rect(p, x, y, w, h)
    rect(p, x, y, 130, h, color, color)
    txt(p, heading, x+350, y+300, w-650, 900, 21, color, True)
    txt(p, body, x+350, y+1370, w-650, h-1550, 17)


def metric(p, value, label, x, y, w, color=BLUE):
    txt(p, value, x, y, w, 1250, 32, color, True, align=1)
    txt(p, label, x, y+1400, w, 1600, 16, DARK, align=1)


def table(p, headers, rows, x=1250, y=3900, widths=None, row_h=2100, fs=17):
    widths = widths or [10300]*len(headers)
    for i, row in enumerate([headers]+rows):
        xx = x
        for j, val in enumerate(row):
            fill = NAVY if i == 0 else (PALE if i % 2 else WHITE)
            rect(p, xx, y+i*row_h, widths[j], row_h, fill, LINE)
            txt(p, val, xx+200, y+i*row_h+230, widths[j]-400, row_h-400,
                fs, WHITE if i == 0 else DARK, i == 0)
            xx += widths[j]


def replay(pattern, second, stem):
    paths = sorted((MAT / "影片").glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one video: {pattern}, found {len(paths)}")
    target = ASSETS / (stem+".png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(second), "-i", str(paths[0]),
                    "-frames:v", "1", "-update", "1", str(target)], check=True)
    return target, paths[0]


def video_link(p, label, video, x, y, w=18000):
    s = txt(p, label, x, y, w, 650, 14, BLUE, True)
    s.String = ""
    field = DOC.createInstance("com.sun.star.text.TextField.URL")
    field.URL = video if isinstance(video, str) and video.startswith("https://") else video.resolve().as_uri()
    field.Representation = label
    cur = s.Text.createTextCursor()
    s.Text.insertTextContent(cur, field, False)
    return field


def build():
    global DOC
    ASSETS.mkdir(parents=True, exist_ok=True)
    frame60, video60 = replay("door_wide_w1f_60_3*成功_側視_實錄重播.mp4", 40.5, "60度開門_實錄重播截圖")
    frame60close, video60close = replay("door_wide_w1f_60_3*成功_近景_實錄重播.mp4", 14.0, "60度開門近景_實錄重播截圖")
    frame48, video48 = replay("door_coopen_core_bench_6*側視_實錄重播.mp4", 48.0, "48度繞行_實錄重播截圖")
    # A presentation-sized crop of the existing timeline: no data or labels are altered.
    fig31 = MAT / "圖表/31_coopen_內側把手串接_名目閉迴路_時間軸.png"
    with Image.open(fig31) as im:
        im.crop((0, 70, im.width, 730)).save(ASSETS / "內側拉關_既有時間軸上半部.png")
    desktop = U.connect_office()
    DOC = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, ())
    DOC.lockControllers()
    pages = DOC.getDrawPages()
    # Impress always retains one page; deleting its last page creates another.
    while pages.getCount() > 1:
        pages.remove(pages.getByIndex(0))

    p = slide("動態環境下移動機械臂\n預測導航與受約束操作", "第七次進度報告 / DRAFT", 
              "老師好，我是陳文顥。這次延續上一期的全身預測控制，從抽屜的直線操作擴展到門的旋轉操作。今天會分開說明已完成的接觸模擬、通過中關門的離線名目結果，以及尚在整合的換抓流程；完整往返任務仍是目標，不是今天已完成的成果。")
    txt(p, "從抽屜操作走向開門、換抓與通過中關門", 1350, 5000, 30000, 1300, 25, BLUE, True)
    txt(p, "Predictive Navigation and Constrained Manipulation\nof Mobile Manipulators in Dynamic Environments", 1350, 7000, 29000, 2300, 20, GRAY)
    txt(p, "報告者：陳文顥\n指導教授：陳榮順 教授", 1350, 11400, 16000, 2300, 20)
    banner(p, "本稿依已保存紀錄整理；完整串接與實跑進度將於正式報告前更新。")

    p = slide("研究目標：導航與操作串成同一趟任務", "01 / 目標與進度", 
              "整體目標是從起點出發，導航到門前，開門後通過，且在整機通過完成以前就有關門進展。接著到櫃體開啟、關閉，再回到起點。開門時不強求一邊穿越，允許先後退拉開，再換抓內側把手。導航段要啟用 GMPC 和導航 CBF；目前開門接觸模擬不是已經驗證完整導航 CBF 串接，也不是新增全身 CBF。", ["reviews/20261027_howard_confirmed_plan.md", "Howard 本輪確認：導航 CBF 開啟、成功開門保留影片、同趟完整往返為目標"])
    steps = [("起點 → 門前", "導航 GMPC ＋導航 CBF", BLUE), ("後退拉開門", "底盤與手臂協同控制", GREEN),
             ("通過途中關門", "解除外側 → 換抓內側", ORANGE), ("抵達櫃體", "導航與操作控制交接", BLUE),
             ("開櫃 → 關櫃", "承接前期抽屜操作能力", BLUE), ("返回起點", "同一趟連續紀錄", ORANGE)]
    for i, (a,b,c) in enumerate(steps):
        x = 1250 + (i%3)*10600
        y = 3900 + (i//3)*5350
        card(p, f"{i+1}  {a}", b, x, y, 10100, 4200, c)
    banner(p, "以上是目標流程；尚未完成同一趟的開門、通行、抽屜操作與往返。", color=ORANGE)

    p = slide("本期進展：從直線抽屜，擴展到旋轉門", "01 / 目標與進度",
              "上一期已展示完整抽屜操作及預測步數比較。這一期的新增內容是門的接觸操作、底盤繞行與換抓承接。必須把三個證據層級分開：60 度單趟開門與受控暫停已通過；約 48 度繞行開門已有接觸證據但整體可靠性未通過；通過中關門目前是離線名目閉迴路。它們不是同一配置或同一趟實驗。", ["第六次講稿.md", "第七次進度報告素材/資料/NOTE_door_wide_w1f_60_3.md", "第七次進度報告素材/規格/COOPEN_核心段接觸模擬_成果與限制.md", "第七次進度報告素材/資料/coopen_inner_wgmpc_v2_summary.json"])
    table(p, ["工作項目", "本期已有證據", "證據界線"], [
        ["前期：抽屜開關與返回", "W-GMPC 與完整抽屜流程", "既有成果，不重複算本期新增"],
        ["拉門開啟與暫停", "約 60°，單趟判定通過", "非穿門；主動保持條件"],
        ["開門同期底盤繞行", "約 47.74°，接觸模擬", "整體判定未通過"],
        ["換抓、通過中拉關", "名目模型推得門角至 2.20°", "離線；尚非物理接觸成果"],
    ], widths=[9000,11200,11100], row_h=2200, fs=16)
    banner(p, "本期研究問題：旋轉機構與接觸切換下，全身預測控制如何持續執行任務？")

    p = slide("控制方法：參考、全身預測與接觸保護分工", "02 / 方法",
              "這裡要清楚說明 GMPC 發揮作用的位置。參考生成器只給目標與引導，不直接寫入理想關節路徑。W-GMPC 使用實測狀態、手臂設定點和偏移估計，聯合求出底盤三維與手臂六維命令。命令再經執行鏈；接觸與幾何監看可拒絕或中止，不把這些監看說成已在 MPC 內完成全身避碰。導航階段另有導航 CBF，不能把它當成目前所有操作段都已有 CBF 保證。", ["src/ammr_wholebody_mpc/ammr_wholebody_mpc/wgmpc_core_sp.py", "evaluation/door_regrasp_mode1.py", "evaluation/door_regrasp_gates3.py", "evaluation/door_regrasp_policy6.py"])
    card(p, "任務與參考", "門鉸鏈／把手座標\nTCP 目標＋底盤引導\n依任務相位切換", 1250, 4200, 9100, 4700)
    txt(p, "→", 10600, 5600, 1100, 1200, 32, BLUE, True)
    card(p, "W-GMPC", "底盤 3 ＋手臂 6 聯合解算\n設定點增廣模型＋延遲補償\n速度／加速度／關節限制", 12000, 4200, 10500, 4700)
    txt(p, "→", 22700, 5600, 1100, 1200, 32, BLUE, True)
    card(p, "執行與回授", "命令來源核對與模式權威\n逐步套用、實測狀態回授\n設定點與偏移估計承接", 24100, 4200, 8500, 4700)
    rect(p, 1250, 10100, 31300, 3900, PALE, LINE)
    txt(p, "幾何／接觸保護（獨立於主要任務追蹤）", 1650, 10400, 29000, 1000, 21, ORANGE, True)
    txt(p, "淨空預測、指定接觸、資料新鮮度、持握漂移與失接觸\n違規時拒絕模式提交或閂鎖中止；不等同全身 CBF 安全保證", 1650, 11700, 29000, 1900, 18)
    banner(p, "GMPC 算的是回授控制命令；IK／路徑搜尋只提供可行參考，不代替控制器。")

    p = slide("模型修正：開門時更新偏移，暫停時凍結", "02 / 方法",
              "延續上一期的設定點模型，不假設速率命令立即變成實際關節速度。這一期發現開門構型改變後，固定偏移的模型誤差會變大，因此在原有有效區間規則下恢復偏移更新，暫停時再凍結。因果影子重播的一區間鉛直殘差 RMS 由 0.776 降到 0.045 毫米，這是預測一致性的證據，不直接等於接觸控制改善。下一頁的實跑比較才展示實際漂移；接觸負載可能被學入，因此承載與內力風險仍要分開觀察。", ["第七次進度報告素材/資料/door_coopen_core_bench_3_offset_shadow.json", "第七次進度報告素材/資料/door_coopen_core_bench_6_compare.json", "第七次進度報告素材/規格/COOPEN_core_spec_draft14.md"])
    txt(p, "手臂逐物理步模型", 1400, 3600, 29000, 900, 23, NAVY, True)
    txt(p, "qₐ,k+1 = qₐ,k + α ⊙ (sₖ − qₐ,k) + α ⊙ d̂ₖ", 1900, 5050, 29200, 1500, 25, BLUE, True)
    txt(p, "sₖ：位置設定點　　α：辨識追蹤係數　　d̂ₖ：區間偏移估計", 1900, 6900, 29200, 1000, 17, GRAY)
    card(p, "開門／收斂期間", "有效實測區間更新 d̂\n延遲預推與 QP 使用同一偏移", 1400, 9000, 14900, 4000, BLUE)
    card(p, "暫停生效後", "凍結估計，不繼續學習\n停止／失效處置仍優先", 17200, 9000, 14900, 4000, ORANGE)
    txt(p, "因果影子重播：鉛直一區間殘差 RMS  0.776 → 0.045 mm", 1500, 14000, 30000, 1000, 22, GREEN, True)
    banner(p, "影子重播是預測核對，非閉迴路與接觸穩定證明；負載學入風險尚未排除。", color=ORANGE)

    p = slide("成果一：60° 開門與受控暫停，單趟通過", "03 / Isaac 接觸模擬",
              "這是本期可以明確展示的成功影片。機器人實際夾持把手，底盤與手臂共同開門至約 60 度，接續收斂、受控暫停與主動保持下的停止觀察，事前凍結的判定器全部通過。這不是穿門成果，也不能推論撤除保持後仍停止、跨條件可靠或實機成功。影片是既有實錄姿態重播，不是原始相機錄影。", ["第七次進度報告素材/資料/NOTE_door_wide_w1f_60_3.md", str(video60.relative_to(ROOT)), str(video60close.relative_to(ROOT))])
    image_fit(p, frame60, 1250, 3450, 20000, 10000)
    image_fit(p, frame60close, 21900, 3450, 10600, 6400)
    metric(p, "60.005°", "進入暫停時的實測門角", 21800, 10800, 10800, GREEN)
    txt(p, "事前判定器通過｜主動保持停止觀察 10.06 s", 1500, 14000, 29000, 1000, 21, GREEN, True)
    video_link(p, "播放：側視實錄重播（需保持素材路徑）", video60, 1400, 15400, 20500)
    video_link(p, "近景重播", video60close, 23100, 15400, 8500)
    banner(p, "單趟、已知場景；非穿門成果。暫停期手指上限 20 N；不代表撤除保持後仍停住。", color=GRAY)

    p = slide("成果二：持握開門時，底盤同步往出口繞行", "03 / Isaac 接觸模擬",
              "第二個成果是指定台架初態下的繞行開門核心段。實際閉爪後，門開到 47.74 度，出口距離指標下降 0.176 公尺，展示底盤不是固定在原地。但本趟整體判定不通過：持握窗有低承載物理步，停止窗也有超標樣本。台架初態繞過自主接近，所以不能寫成自主抓取或完整通行成功。它也與前頁 60 度成功趟不同抓法、不同配置。", ["第七次進度報告素材/資料/bench_6_door_coopen_check_core12.json", "第七次進度報告素材/規格/COOPEN_核心段接觸模擬_成果與限制.md", str(video48.relative_to(ROOT))])
    image_fit(p, frame48, 1250, 3500, 21500, 11000)
    metric(p, "47.74°", "OPEN 實測最大門角\n參考 48°", 23000, 3700, 9300)
    metric(p, "0.176 m", "出口距離指標下降\n不是已穿越距離", 23000, 8200, 9300)
    txt(p, "指定初態＋實際閉爪取證；非自主接近", 1400, 14600, 29000, 850, 18, GRAY)
    video_link(p, "播放：側視實錄重播（整體判定未通過）", video48, 1400, 15500, 29000)
    banner(p, "整體判定未通過：29 個低承載物理步、停止十秒窗內 2 個手臂超標樣本。", color=RED, fill=0xFCF1ED)

    p = slide("實測比較：共同門角區間的持握漂移較小", "03 / 模型修正的實測證據",
              "圖中灰色是開門早期即凍結偏移估計的趟次，另外三條為開門期間恢復更新的配置。共同門角 40 到 43.7 度，平均持握漂移由 4.48 毫米變成 1.33、1.32、1.35 毫米，三趟分箱曲線相近。不把不同診斷與暫停配置混稱嚴格單變因因果。末段仍回升到 4.82 毫米，離 5 毫米門檻的 0.18 毫米只是名目差額，不能稱抗擾動餘裕。", ["第七次進度報告素材/圖表/29_coopen_核心段接觸模擬_漂移四趟比較.png", "第七次進度報告素材/資料/door_coopen_core_bench_6_compare.json"])
    image_fit(p, MAT/"圖表/29_coopen_核心段接觸模擬_漂移四趟比較.png", 1250, 3600, 31300, 10300)
    txt(p, "40–43.7° 平均漂移：4.48 → 1.33／1.32／1.35 mm", 1500, 14000, 30000, 1100, 23, BLUE, True)
    banner(p, "描述性重現，非嚴格單變因因果；末段最大 4.82 mm，可靠性仍未通過。", color=ORANGE)

    p = slide("通過中關門：由外側把手切換到內側把手", "04 / 通行與換抓方法",
              "為避免把手臂一直纏在門扇外側，這條路線允許先解除外側持握，退臂後移到門洞入口，再抓內側把手。雙面把手已建立版本化幾何與資產，但同型同高對稱是建模假設，不能當成不同實門都適用。這一頁是方法骨架，不是實跑動畫。狀態與命令來源要承接，不使用瞬移或理想固定連接來假裝換抓成功。", ["evaluation/door_asset_v3.py", "evaluation/door_regrasp_ref_registry.py", "evaluation/door_regrasp_mode1.py", "evaluation/door_regrasp_exec_core.py"])
    stages = [("外側開門", "先拉開門並受控保持"), ("解除與退開", "張爪，離開外側橫桿"),
              ("移到門洞", "底盤與手臂共同退臂／移動"), ("抓內側把手", "到位准入後才閉爪"),
              ("通過中拉關", "整機清空前已有關門進展")]
    for i,(a,b) in enumerate(stages):
        xx = 1250+i*6350
        card(p, str(i+1)+"  "+a, b, xx, 4200, 5900, 5400, BLUE if i<3 else ORANGE)
        if i<4:
            txt(p, "→", xx+5900, 6100, 450, 900, 18, BLUE, True)
    txt(p, "換抓交接的三個必要條件", 1450, 10800, 29000, 1100, 23, NAVY, True)
    txt(p, "① 前段已確認停止　② 新模式有有效接觸／幾何證據\n③ 設定點、偏移估計與命令來源持續承接，不重新初始化", 1450, 12200, 29000, 2100, 19)
    banner(p, "雙面把手為明示假設；新換抓與通過中關門尚未完成 Isaac 接觸實跑。", color=ORANGE)

    p = slide("離線名目結果：通過途中已有約 45.7° 關門進展", "04 / 離線模型・不是接觸實跑",
              "這組結果使用真正的 W-GMPC 求解器與辨識設定點受控體、指定延遲，從已保存的開門末態開始。門角在抓持段由 TCP 推得，不是門動力學量測。模型中，門由 47.89 度關至 2.20 度，底盤越過圓盤清空線前已關約 45.7 度，整機其後也越過遠側面。取樣自檢沒有所列違規，但自由段 TCP 最大落後 120 毫米，仍有關節飽和，而且未驗接觸力、E2、安全層、完全關閉及 Isaac。此為早期完整拉關模型，與下一頁最新換抓參考不是同一版，不能拼成同趟成功。", ["第七次進度報告素材/資料/coopen_inner_wgmpc_v2_summary.json", "第七次進度報告素材/圖表/31_coopen_內側把手串接_名目閉迴路_時間軸.png", "第七次進度報告素材/圖表/32_coopen_內側把手串接_名目閉迴路_俯視.png"])
    image_fit(p, ASSETS/"內側拉關_既有時間軸上半部.png", 1250, 3600, 21000, 10000)
    metric(p, "47.89° → 2.20°", "模型推得門角\n不是門關節實測", 22600, 3700, 10000)
    metric(p, "45.7°", "底盤圓盤清空前\n已發生的關門進展", 22600, 8400, 10000)
    txt(p, "20 Hz 取樣自檢無所列違規；自由段 TCP 最大落後 120 mm。", 1400, 14400, 30000, 1200, 19, BLUE, True)
    banner(p, "理想名目初始化、無接觸力／門動力學／E2；未驗最後 2.2°→0°，不是完整通行實證。", color=ORANGE)

    p = slide("目前整合：真控制器承接實錄末態，完成換抓名目追蹤", "04 / 最新換抓模型與待驗項",
              "最新換抓版本承接 bench_6 實錄的關節、設定點、套用命令與模型偏移，不再用理想平衡設定點初始化。內側抓法改為沿橫桿軸轉 30 度，配合經預測淨空核對的時間表。名目模型中，真 W-GMPC 走完退開、收臂移動與內側接近，沒有取樣 stop 或 warn，仍沒有物理接觸力確認。這個版本只到換抓，不包含前頁那個完整拉關結果。接下來是實際三節點與執行端的完整啟動和換抓聯測，完成後再提出單趟 Isaac 申請，不把準備工作算成任務成功。", ["evaluation/results/door/door_regrasp_ref_v4_wgmpc_v2.json", "evaluation/results/door/door_regrasp_ref_v4.json", "reviews/20261028_145143_request.md", "reviews/20261028_152918_request.md"])
    table(p, ["驗證層級", "目前結果", "還缺什麼"], [
        ["新抓法與時間表", "φ＋30°、三個門角參考核對", "有限取樣，不是連續安全證明"],
        ["真 W-GMPC 名目追蹤", "走到內側保持；stop／warn 0", "無接觸力；部分關節飽和"],
        ["ROS 與模式權威", "完整啟動與換抓聯測進行中", "不得提前提交／放手／結束"],
        ["下一個實跑里程碑", "同趟：開門 → 解除 → 內側換抓", "實際閉爪、連續 2 s 保持取證"],
    ], widths=[8500,11600,11200], row_h=2200, fs=16)
    banner(p, "僅名目：收臂 TCP 落後達 120 mm；0.29 s 壓力核對未過。新換抓不含拉關。", color=ORANGE)

    p = slide("結論與下一步：先接成一趟，再完成往返", "05 / 結論與後續工作",
              "目前可報告三層成果：第一，單趟 60 度開門與主動保持停止通過；第二，指定初態下，接觸模擬展示開門同期往出口繞行及持握漂移改善，但可靠性未通過；第三，通過中關門的離線名目路線已建立，新換抓正在整合。接下來優先取得同一趟的開門到換抓接觸證據，再接通過中關門，最後串接導航、開關抽屜與返回。這是研究目標，不承諾報告前一定完成；正式報告按實際取得的證據更新。成功開門影片已保留，後續成功子段也會保留，完整同趟紀錄另外驗證。", ["第七次進度報告素材/規格/COOPEN_核心段接觸模擬_成果與限制.md", "reviews/20261027_howard_confirmed_plan.md", "reviews/20261028_152918_request.md"])
    card(p, "已取得", "60° 開門單趟通過\n約 48° 繞行開門接觸證據\n換抓／拉關離線名目結果", 1250, 3800, 15000, 5600, GREEN)
    card(p, "尚未取得", "同趟自主開門與內側換抓\n物理通行、關門與退出\n導航＋抽屜＋返回完整串接", 17300, 3800, 15000, 5600, ORANGE)
    txt(p, "接下來的交付順序", 1450, 10500, 29000, 1000, 24, NAVY, True)
    txt(p, "1  同趟開門 → 內側換抓接觸驗證＋影片\n2  通過中拉關 → 放手退出 → 控制交還導航\n3  導航 GMPC＋導航 CBF → 開關抽屜 → 返回起點", 1450, 11900, 30000, 3400, 20)
    banner(p, "目標是完整流程；正式成果只報已實際取得的層級，不把子段或不同趟拼成成功。")

    p = slide("備用：視覺預抓取已接入，但不能直接放行閉爪", "附錄 A / 本期其他進展",
              "視覺支線目前完成限定預抓取。已知把手與靜態先驗下，以幾何視覺鎖定目標，到位誤差相對鎖定目標 1.83 毫米，保持 0.51 秒；相對真值的 TCP 誤差約 3.99 毫米。估計與控制誤差會同方向疊加，這比只看控制到位更重要。像素中心契約候選版在 82 格共同工具目標上降低接近軸絕對誤差中位數，但仍有變差影格，未建立後續抓取可接受誤差界，不能宣稱視覺抓取或泛化完成。", ["第七次進度報告素材/規格/DL3_run_result.yaml", "第七次進度報告素材/規格/PC2_result.yaml"])
    image_fit(p, MAT/"圖表/16_PC2_像素契約新版對照_誤差與餘裕.png", 1250, 3500, 20000, 11000)
    metric(p, "1.83 mm", "相對視覺鎖定目標\n預抓取到位誤差", 22000, 3900, 10000)
    metric(p, "1.36 → 0.97 mm", "82 格共同工具目標\n接近軸絕對誤差中位", 22000, 8800, 10000)
    banner(p, "僅預抓取；未閉爪／操作。候選像素契約離線改善，不等於已有抓取誤差保證。", color=ORANGE)

    p = slide("備用：通過中拉關的空間路線", "附錄 B / 離線名目模型",
              "這張是早期內側把手完整拉關名目模型的俯視圖。底盤圓盤清空與整機清空是不同條件；必須連手臂與夾爪都退出牆面包絡。路線來自對稱雙側把手假設與名目模型，不是實際 Isaac 影片，也不是最新 φ＋30 度換抓參考。", ["第七次進度報告素材/圖表/32_coopen_內側把手串接_名目閉迴路_俯視.png", "第七次進度報告素材/資料/coopen_inner_wgmpc_v2_summary.json"])
    image_fit(p, MAT/"圖表/32_coopen_內側把手串接_名目閉迴路_俯視.png", 1250, 3200, 31300, 12500)
    banner(p, "舊版完整拉關模型：非 Isaac；非最新換抓參考。底盤過牆不等於整機清空。", color=ORANGE)

    p = slide("備用：為什麼流程 DONE，仍可能判定不通過？", "附錄 C / 判定與證據界線",
              "這頁用來回答為什麼影片上流程結束，卻不說任務可靠成功。bench_6 線上狀態機走完，但逐物理步獨立重算發現兩類違規。29 個物理步低承載，最長八步是兩指交替，不是某一指連續八步，也不能直接說成滑脫。停止十秒窗內有兩個超標樣本；已記錄的時序支持取樣差異，但監督器當時窗口未完整保存，不能宣稱原因完全確認。正式判定不改。", ["第七次進度報告素材/資料/bench_6_door_coopen_check_core12.json", "第七次進度報告素材/資料/door_coopen_core_bench_6_reconcile.json", "第七次進度報告素材/規格/COOPEN_核心段接觸模擬_成果與限制.md"])
    table(p, ["判定項目", "bench_6 實際紀錄", "解讀"], [
        ["低承載", "任一指 < 0.5 N：29 物理步\n最長連續事件 8 步（0.08 s）", "未達線上連續失接觸期限\n但不能稱夾持穩定"],
        ["停止觀察", "十秒窗內 2 樣本 > 0.001 rad/s\n最大約 0.00109 rad/s", "線上與逐步核對結果不同\n不降低取樣率、不放寬門檻"],
        ["末段持握漂移", "最大 4.82 mm／門檻 5 mm", "0.18 mm 是名目差額\n不是已證明的抗擾動餘裕"],
        ["整體結論", "任務流程 DONE；chain_pass False", "保留成果，也保留未通過"],
    ], widths=[7400,13200,10700], row_h=2200, fs=16)
    banner(p, "研究結果以量測與判準為準，不以影片看起來完成或狀態機 DONE 代替。")

    DOC.unlockControllers()
    for ext, filt, method in [("pptx", "Impress MS PowerPoint 2007 XML", DOC.storeAsURL),
                              ("pdf", "impress_pdf_Export", DOC.storeToURL)]:
        target = OUT/f"第七次進度報告_工作稿_v1.{ext}"
        method(uno.systemPathToFileUrl(str(target)), (U.pv("FilterName", filt), U.pv("Overwrite", True)))
        print(target)
    DOC.close(True)
    desktop.terminate()
    lines = ["# 第七次進度報告｜逐頁講稿與來源（工作稿 v1）", "",
             "整理日：2026-10-28。主簡報 12 頁＋備用 3 頁；正式報告日期待填。",
             "進度快照：r66。報告前依實跑更新，不自動把準備完成當成果。",
             "影片超連結指向原有素材；移動整份資料時需重新確認路徑。", ""]
    for s in SLIDES:
        lines += [f"## {s['n']:02d}｜{s['title'].replace(chr(10), ' ')}", "", s['notes'], "", "來源：", ""]
        lines += [f"- [{x}]({(ROOT/x).as_posix()})" for x in s['sources']]
        lines += [""]
    (OUT/"逐頁講稿與來源_v1.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT/"投影片索引_v1.json").write_text(json.dumps(SLIDES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {len(SLIDES)} slides (12 main + 3 appendix).")


def preview_and_audit():
    preview = OUT / "預覽"
    preview.mkdir(exist_ok=True)
    subprocess.run(["pdftoppm", "-scale-to", "1100", "-png",
                    str(OUT/"第七次進度報告_工作稿_v1.pdf"), str(preview/"slide")], check=True)
    files = sorted(preview.glob("slide-*.png"))
    sheet = Image.new("RGB", (1320, 5*283), "#e8edf3")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    for i, file in enumerate(files):
        im = Image.open(file).convert("RGB")
        im.thumbnail((420, 237))
        x, y = 10+(i%3)*440, 32+(i//3)*283
        sheet.paste(im, (x,y))
        draw.text((x,y-25), f"{i+1:02d}"+(" / appendix" if i>=12 else ""), font=font, fill="#17365d")
    sheet.save(OUT/"預覽總覽_v1.png")
    ns = {"a":"http://schemas.openxmlformats.org/drawingml/2006/main"}
    with zipfile.ZipFile(OUT/"第七次進度報告_工作稿_v1.pptx") as z:
        texts = ["".join(ET.fromstring(z.read(f"ppt/slides/slide{i}.xml")).itertext()) for i in range(1,16)]
        assert all(len(t)>60 for t in texts), "A slide is missing its text"
        notes = [n for n in z.namelist() if n.startswith("ppt/notesSlides/notesSlide") and n.endswith(".xml")]
        assert len(notes)==15, f"Expected speaker notes on all 15 slides, got {len(notes)}"
    index = json.loads((OUT/"投影片索引_v1.json").read_text(encoding="utf-8"))
    missing = sorted({s for row in index for s in row['sources']
                      if not s.startswith("Howard ") and not (ROOT/s).is_file()})
    assert not missing, f"Missing source files: {missing}"
    assert len(files)==15
    print("Checked: 15 rendered pages, slide text, 15 speaker-note pages, all source-file paths.")


if __name__ == "__main__":
    if "--preview-only" not in sys.argv:
        build()
    preview_and_audit()
