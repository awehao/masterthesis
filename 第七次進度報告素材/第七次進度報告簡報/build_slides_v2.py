#!/usr/bin/env python3
"""Report seven v2: continue from the drawer, then related work and evidence.

Only creates presentation artifacts; never starts a controller or simulator.
Uses local report helpers and citations verified on 2026-10-28.
"""
from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import uno
from PIL import Image, ImageDraw, ImageFont

import build_slides as B

ROOT, OUT, MAT = B.ROOT, B.OUT, B.MAT
V6 = ROOT / "第六次進度報告素材"
B.DRAFT_VERSION = "v2"
STEM = "第七次進度報告_從抽屜到門操作_v2"
R1 = "https://wbmpc.github.io/static/pdfs/paper.pdf"
R2 = "https://arxiv.org/html/2608.00206v1"
R3 = "https://arxiv.org/html/2410.22910v1"
R4 = "https://khu.elsevierpure.com/en/publications/motion-planning-of-mobile-manipulator-for-navigation-including-do/"
R5 = "https://arxiv.org/abs/2106.04202"
R6 = "https://arxiv.org/abs/2403.07317"


def S(title, sec, note, sources=()):
    return B.slide(title, sec, note, sources)


def refline(p, label):
    B.txt(p, label, 1450, 15050, 30000, 780, 11, B.GRAY)


def build():
    B.SLIDES.clear()
    desktop = B.U.connect_office()
    B.DOC = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, ())
    B.DOC.lockControllers()
    pages = B.DOC.getDrawPages()
    while pages.getCount() > 1:
        pages.remove(pages.getByIndex(0))
    assets = OUT / "素材_v2"
    assets.mkdir(exist_ok=True)
    drawer_video = V6/"影片/05_MotM抽屜開關_全景_3倍速.mp4"
    drawer_frame = assets/"前期抽屜操作_既有影片截圖.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "18", "-i", str(drawer_video),
                    "-frames:v", "1", "-update", "1", str(drawer_frame)], check=True)
    video60 = next((MAT/"影片").glob("door_wide_w1f_60_3*成功_側視_實錄重播.mp4"))
    video48 = next((MAT/"影片").glob("door_coopen_core_bench_6*側視_實錄重播.mp4"))
    evidence = "第七次進度報告素材/第七次進度報告簡報/文獻差異與改善證據_v2.md"
    drawer_sources = ["第六次講稿.md", "第六次進度報告素材/簡報用實驗比較表.md",
                      "evaluation/results/motm_speed/park_hold_formal_summary.yaml"]

    p = S("由抽屜操作到門通行：\n面向位置伺服介面的全身預測控制", "第七次進度報告 / 研究進展",
          "老師好，我是陳文顥。上次我用 W-GMPC 完成抽屜開啟、關閉與返回，並做整機協調和預測步數比較。這次不是只增加一個開門展示，而是研究同一套位置伺服型全身控制，在旋轉機構、構型變化及接觸切換下，如何維持預測和實際執行的一致性。報告會先從抽屜接續，再對照相關方法，最後用目前已有數字說明改善和限制。", drawer_sources+[evidence])
    B.txt(p, "動態環境下移動機械臂預測導航與受約束操作", 1400, 4750, 30000, 1050, 21, B.BLUE)
    B.txt(p, "上次：協同開關抽屜\n本次：模型更新、鉸鏈操作與換抓承接", 1400, 6850, 29700, 3100, 26, B.NAVY, True)
    B.txt(p, "報告者：陳文顥\n指導教授：陳榮順 教授", 1400, 11400, 21000, 2300, 20)
    B.banner(p, "報告主軸：解決什麼執行問題 → 和既有方法有何不同 → 目前證據改善了什麼。")

    p = S("上次已完成：W-GMPC 協同開關抽屜並返回", "01 / 從第六次成果接續",
          "先從上次已完成的地方開始。W-GMPC 同時決定底盤和手臂命令，完成導航、把手夾持、開啟二百毫米、關閉、釋放與返回。在嚴格停車對照的三對實驗中，兩組皆通過，整機協調的平均完整時間由九十三點六二秒降到八十點七二秒。這是兩套完整策略比較，主要時間收益在入場與交棒，不能說所有收益都是底盤移動造成。本頁影片是既有流程展示，數據另來自正式配對批次。", drawer_sources+[str(drawer_video.relative_to(ROOT))])
    B.image_fit(p, drawer_frame, 1250, 3450, 20500, 10600)
    B.metric(p, "200 mm", "開啟 → 保持 → 關閉\n解除 → 返回起點", 22300, 3800, 10000)
    B.metric(p, "93.62 → 80.72 s", "平均完整任務時間\n三對策略比較", 22000, 8500, 10400, B.GREEN)
    B.txt(p, "兩組皆 3/3 通過；逐對時間縮短 13.38–14.05%。", 1450, 14200, 30000, 1000, 20, B.GREEN, True)
    B.video_link(p, "播放上次抽屜流程（3 倍速；示範影片與正式配對批次分開）", drawer_video, 1450, 15400, 30000)
    B.banner(p, "前期已證據化的改善：完整策略較快；不是本期新實驗，也不是對外部論文的勝負。", color=B.GRAY)

    p = S("接下來的問題：抽屜追得住，不代表門也能操作", "01 / 本期要解決的問題",
          "抽屜與門都只有一個機構自由度，但控制難度不同。抽屜沿固定方向，門的把手位置與姿態會隨門角改變。當底盤繞過門扇，手臂構型也持續變化，固定偏移模型可能不再準確。更重要的是，開到角度不代表能放手、換抓或通過。因此本期不只追求更大的門角，而是把問題放在預測一致性、可執行參考與接觸切換承接。", ["第六次講稿.md", "開門實作_文獻參考與研究改善定位.md", evidence])
    B.table(p, ["", "抽屜：前期已完成", "門：本期新增問題"], [
        ["機構與目標", "沿滑軌平移；方向較固定", "鉸鏈旋轉；位置與姿態一起改變"],
        ["全身構型", "在既定操作區協同開關", "底盤繞門，手臂可達與自碰更敏感"],
        ["模型落差", "設定點追蹤與偏移補償", "構型／接觸改變後，固定估計是否仍有效？"],
        ["完成條件", "關回、放開、退出、返回", "開門之外，還要換抓、通過與拉關"],
    ], widths=[5500,11700,14100], row_h=2200, fs=17)
    B.banner(p, "研究問題：如何讓同一套 W-GMPC，從單一操作延伸到可執行的連續操作？")

    p = S("相關研究已做到什麼？本研究的切入點在哪裡？", "02 / 方法差異，不把不同寫成較好",
          "這裡選三個直接相關的方法。Wang 等人的重點是全身 MPC 的任務優先序轉換，已展示實機自動回關門的開啟與通過；不是只能停車操作。Sinaei 等人在較低維的機器人與門耦合規劃中，以軟工作空間條件描述可達性，手臂交給 IK。Du 等人已有混合位置與速度介面的雙層 MPC，所以我們也不能宣稱首次處理這種介面。我的切入點是這台全向底盤加單臂，明確預測位置設定點的執行，並在機構操作中研究模型更新與控制承接。這是方法與假設比較，不是已重現外部方法的效能排行榜。", [R1,R2,R3,evidence])
    B.table(p, ["相關方法", "主要設計／已有能力", "本研究對應的差異"], [
        ["Wang et al.\nICRA 2025 [R1]", "任務優先序轉換的全身 MPC\n已有實機開門、擋門、通過", "顯式預測位置設定點的執行\n研究操作中的偏移更新與承接"],
        ["Sinaei et al.\n2026 預印本 [R2]", "低維機器人—門耦合 NMPC\n軟可達性＋下游手臂 IK", "保留六軸構型、設定點與限位\n另核取樣淨空；門動力學較簡化"],
        ["Du et al.\n2024 v1 [R3]", "雙層 Bézier MPC＋預測導納\n雙臂、混合位置／速度介面", "單臂設定點增廣＋區間估計\n沿用既有 SQP/QP 全身核心"],
    ], widths=[6200,12600,12500], row_h=2650, fs=16)
    refline(p, "[R1] 作者原文 §III／IV　[R2] arXiv:2608.00206v1 §1–2　[R3] arXiv:2410.22910v1 §III")
    B.banner(p, "文獻已有全身 MPC、門通行與混合介面；本研究不以這些能力存在本身宣稱首創。")

    p = S("本研究的切入：讓預測模型貼近實際執行鏈", "02 / 面向位置伺服介面的設計",
          "我要處理的不是別人都不會開門，而是位置伺服平台上的執行一致性。控制命令先積分為位置設定點，實際角度再追它；模型若只看理想關節速率，會忽略已累積的設定點與追蹤落後。這部分模型上次已建立，本次沿著三個問題推進：開門時估計是否需要更新、幾何可行參考是否仍可被控制器追蹤，以及換抓時如何不重設控制狀態。設定點增廣與自適應本身不是新概念，本研究主張必須落在具體實作與對照證據。", ["src/ammr_wholebody_mpc/ammr_wholebody_mpc/wgmpc_core_sp.py", "文獻對照_第二階段_WGMPC.md", evidence])
    B.card(p, "命令", "底盤速度＋手臂速率\n同一最佳化聯合決定", 1250, 3700, 9200, 3300)
    B.txt(p, "→", 10600, 4600, 1100, 1100, 30, B.BLUE, True)
    B.card(p, "執行狀態", "速率先積分成設定點 s\n位置伺服追蹤實際角度 q", 12000, 3700, 10800, 3300)
    B.txt(p, "→", 22900, 4600, 1100, 1100, 30, B.BLUE, True)
    B.card(p, "實測回授", "量測 q、s、已套用命令\n修正下一輪預測", 24200, 3700, 8350, 3300)
    for i,(a,b) in enumerate([
        ("1  模型一致", "設定點動態＋延遲補償\n不把命令當成已發生的運動"),
        ("2  操作中更新", "有效區間估計偏移\n處理構型改變時的預測落差"),
        ("3  階段間承接", "解除、移動、再抓取\n保留設定點與命令歷史")]):
        B.card(p,a,b,1250+i*10600,8900,10100,4600,B.BLUE if i==0 else B.ORANGE)
    B.banner(p, "設計重點是執行一致性；是否更好，要看本平台直接對照，不能只靠架構名稱。")

    p = S("控制模型沿用上次：9 維命令，15 維預測狀態", "03 / W-GMPC 如何發揮作用",
          "這頁只回顧必要模型，不重新講完整推導。底盤三維加手臂六維構成九維實際構型，再把六維設定點加入，得到十五維預測狀態，控制輸入仍是九維。每輪 W-GMPC 聯合產生底盤與手臂命令，再由執行鏈套用。幾何監看位於外部，不能說所有避碰都已在 QP 內。導航沿用 GMPC 與導航 CBF，操作核心是目前實作的 W-GMPC；後者不能因名字相近就宣稱直接繼承 Tang 的全部李群誤差狀態理論。", ["第六次講稿.md", "src/ammr_wholebody_mpc/ammr_wholebody_mpc/wgmpc_core_sp.py", "文獻對照_第二階段_WGMPC.md", R6])
    B.txt(p, "預測狀態 z = [ q ; s ] ∈ ℝ¹⁵　　控制命令 u ∈ ℝ⁹", 1550, 3650, 30000, 1400, 24, B.NAVY, True)
    B.txt(p, "qₐ,k+1 = qₐ,k + α ⊙ (sₖ − qₐ,k) + α ⊙ d̂ₖ", 1550, 5600, 30000, 1400, 25, B.BLUE, True)
    B.txt(p, "sₖ+1 = sₖ + Δtₚ uₐ,k", 1550, 7500, 30000, 1250, 25, B.BLUE, True)
    B.txt(p, "qₐ：實測角度　s：位置設定點　α：辨識係數　d̂：估計偏移", 1550, 9150, 30000, 1000, 17, B.GRAY)
    B.card(p, "求解器內", "TCP 位姿追蹤、協同引導\n關節／設定點限位、命令速度與增量", 1400, 11100, 15000, 3400)
    B.card(p, "執行與外部監看", "實際命令承接、指定接觸與淨空\n資料新鮮度、持握漂移、中止", 17400, 11100, 15000, 3400, B.ORANGE)
    B.banner(p, "本期新增重點：模型偏移何時更新、參考能否執行、接觸切換如何承接。")

    p = S("本期改進一：開門期間更新偏移，而非全程凍結", "03 / 由觀察問題到控制修正",
          "初期開門結果在末段出現位置漂移，預測起點另有鉛直偏差。我沒有直接提高漂移門檻，而是對既有紀錄做因果影子重播：兩版本同一起點，每個區間先預測再更新，恢復更新後的一區間鉛直殘差 RMS 從零點七七六降到零點零四五毫米。據此把既有估計規則延伸到開門與收斂期間，暫停生效才凍結。這是模型一致性的改善證據，不表示已分離所有接觸負載原因，也不是穩定性定理。", ["第七次進度報告素材/資料/door_coopen_core_bench_3_offset_shadow.json", "第七次進度報告素材/規格/COOPEN_core_spec_draft14.md"])
    B.card(p,"原本","切入開門運動後凍結 d̂\n約 100 s 不再更新\n末段持握位置誤差增大",1250,3900,15000,5000,B.GRAY)
    B.card(p,"本期候選","開門／收斂期間按有效區間更新\n暫停生效時才凍結\n停止與失效處置仍優先",17300,3900,15000,5000,B.BLUE)
    B.metric(p,"0.776 → 0.045 mm","既有紀錄的因果影子重播\n鉛直一區間預測殘差 RMS",2900,10700,28000,B.GREEN)
    B.banner(p,"更新前估計做預測，不用未來資料回填；仍須分報接觸承載，避免把負載學入當成成功。",color=B.ORANGE)

    p = S("實際改善：共同門角區間的持握漂移下降", "03 / 預測修正的接觸模擬結果",
          "接著看物理接觸模擬，而不只看影子重播。灰色為開門早期凍結，另外三條是開門更新的趟次。共同四十到四十三點七度，持握平均漂移由四點四八毫米變為一點三三、一點三二、一點三五毫米，曲線三趟相近。這是可報告的量測改善；但診斷、GC 與暫停設定不完全相同，不稱嚴格單變因因果。末段仍回升至四點八二毫米，尚有薄弱處。", ["第七次進度報告素材/圖表/29_coopen_核心段接觸模擬_漂移四趟比較.png", "第七次進度報告素材/資料/door_coopen_core_bench_6_compare.json"])
    B.image_fit(p,MAT/"圖表/29_coopen_核心段接觸模擬_漂移四趟比較.png",1250,3400,31300,10400)
    B.txt(p,"40–43.7° 平均：4.48 → 1.33／1.32／1.35 mm",1450,14000,30000,1150,24,B.BLUE,True)
    B.banner(p,"相較本平台早期配置的改善，非超越外部論文；末段最大 4.82 mm，可靠性仍未通過。",color=B.ORANGE)

    p = S("任務能力擴展：單趟 60° 開門與受控暫停通過", "03 / 從直線機構到鉸鏈機構",
          "從抽屜延伸到門，已有一項可清楚展示的接觸模擬成果：實際夾持把手開門至約六十度，完成收斂、受控暫停與十點零六秒主動保持停止觀察，事前判定器通過。這是一趟功能成果，不是可和論文跨平台比較的成功率，也不是穿門。畫面為既有實錄姿態重播，並未重跑控制。", ["第七次進度報告素材/資料/NOTE_door_wide_w1f_60_3.md",str(video60.relative_to(ROOT))])
    B.image_fit(p,OUT/"素材/60度開門_實錄重播截圖.png",1250,3450,20000,10000)
    B.image_fit(p,OUT/"素材/60度開門近景_實錄重播截圖.png",21900,3450,10600,6400)
    B.metric(p,"60.005°","進入暫停時的實測門角",21800,10800,10800,B.GREEN)
    B.txt(p,"事前判定器通過｜主動保持停止觀察 10.06 s",1450,14000,30000,1000,21,B.GREEN,True)
    B.video_link(p,"播放：60° 開門側視實錄重播",video60,1450,15400,29000)
    B.banner(p,"已知場景、單趟、暫停期 20 N 上限；不代表撤除保持後停止、實機或通行能力。",color=B.GRAY)

    p = S("全身協同證據：開門同期，底盤已往出口繞行", "03 / 核心動作證據與未通過項分開",
          "另一個繞緣抓法的指定初態實驗，實際開門至四十七點七四度，出口距離指標下降零點一七六公尺，顯示底盤與手臂共同執行開門及繞行。不過整體判定仍未通過，原因是低承載物理步與獨立停止窗超標。這裡強調的是新增核心動作證據，不把流程 DONE 說成可靠成功；本趟也不是自主接近，且和六十度成功趟並非同一抓法配置。", ["第七次進度報告素材/規格/COOPEN_核心段接觸模擬_成果與限制.md", "第七次進度報告素材/資料/bench_6_door_coopen_check_core12.json",str(video48.relative_to(ROOT))])
    B.image_fit(p,OUT/"素材/48度繞行_實錄重播截圖.png",1250,3500,21500,10800)
    B.metric(p,"47.74°","OPEN 實測最大門角\n參考 48°",23000,3700,9300)
    B.metric(p,"0.176 m","出口距離指標下降\n不是已通過門洞的距離",23000,8200,9300)
    B.txt(p,"指定台架初態＋實際閉爪取證；尚非自主接近或通行。",1450,14400,30000,900,18,B.GRAY)
    B.video_link(p,"播放：繞行開門實錄重播（整體判定未通過）",video48,1450,15500,30000)
    B.banner(p,"未通過保留：29 個低承載物理步、停止十秒窗內 2 個手臂超標樣本。",color=B.RED,fill=0xFCF1ED)

    p = S("本期改進二：參考可執行性，不只看 IK 是否有解", "04 / 從幾何路徑到回授執行",
          "第二項工作是參考和控制執行的一致性。離線幾何可行並不保證位置伺服加延遲的 W-GMPC 能追上。我們在同一名目設定下比較兩種自由段速度參考：較快參考在實際模型的退臂移動段出現自碰淨空不足；降低速度的版本完成取樣自檢。這只有兩組，不是最適速度搜尋，也不證明一半速度是必要條件。後續換抓規劃再把監看所需扣減納入參考時間化，使幾何門檻與預測監看一致；不靠放寬門檻去湊通過。", ["evaluation/results/door/door_coopen_inner_wgmpc_v1_static.json", "第七次進度報告素材/資料/coopen_inner_wgmpc_v2_summary.json", "evaluation/results/door/door_regrasp_ref_v4_monitor_check_v4.json"])
    B.table(p,["參考／執行條件","名目 W-GMPC 結果","可得到的結論"],[
        ["自由段速度比例 0.995","退臂 → 移動出現淨空違規\nTCP 落後最大約 316 mm","幾何可行，不等於執行可行"],
        ["自由段速度比例 0.5","所列 20 Hz 取樣自檢 0 違規\n自由段仍可落後 120 mm","保留控制餘裕後，找到可執行名目序列"],
        ["後續換抓參考","把預測淨空扣減納入規劃／時間化\n模式切換與准入使用實測證據","讓規劃、監看與提交條件一致"],
    ],widths=[8000,11600,11700],row_h=2800,fs=16)
    B.banner(p,"改善對象是本平台參考／執行配置；不是新規劃理論，也不是已證明物理換抓穩定。",color=B.ORANGE)

    p = S("離線能力：底盤通過途中，模型門角已明顯減少", "04 / 通過中拉關的名目閉迴路",
          "早期完整拉關模型由保存的開門末態開始，包含放手、退臂、移動、內側幾何准入、通過中拉關與退出。模型推得門角由四十七點八九度降到二點二度；在底盤圓盤清空前已有約四十五點七度的關門進展，整機之後也清空。這支持路線在所列名目模型下有執行候選，並非完成物理關門。門角由 TCP 推得，未驗接觸力、門動力學、E2、安全層或最後二點二度。最新換抓已改抓法與參考，不能和這組拼成同趟成功。", ["第七次進度報告素材/資料/coopen_inner_wgmpc_v2_summary.json", "第七次進度報告素材/圖表/31_coopen_內側把手串接_名目閉迴路_時間軸.png"])
    B.image_fit(p,OUT/"素材/內側拉關_既有時間軸上半部.png",1250,3600,21000,10000)
    B.metric(p,"47.89° → 2.20°","模型推得門角\n非門關節實測",22600,3700,10000)
    B.metric(p,"45.7°","底盤圓盤清空前\n已發生的關門進展",22600,8400,10000)
    B.txt(p,"已取得：可追蹤的名目序列。待取得：同趟換抓、通行與關門的接觸實證。",1450,14300,30000,1350,19,B.BLUE,True)
    B.banner(p,"此圖為舊版完整拉關模型；新換抓模型只到內側保持，兩者不併成一個通過結果。",color=B.ORANGE)

    p = S("改善了什麼？用本平台基線與數字回答", "05 / 貢獻主張與證據",
          "這張把改善分成三個層次。前期完整抽屜策略時間較短，前期五步設定的追蹤和命令反轉率較好；本期影子更新的預測殘差較小，接觸模擬共同門角區間的持握漂移也較小。這些都是相對於本平台的基線，不是對外部論文的直接優越證明。我的方法定位是面向位置伺服型移動機械臂的執行一致性設計與分層實證。設定點模型與更新機制可說清楚，但目前不能直接宣稱顯式設定點模型比所有理想速度模型更好，因為還缺那個直接對照。", drawer_sources+["evaluation/results/horizon_ablation/c0b1_results.yaml", "第七次進度報告素材/資料/door_coopen_core_bench_3_offset_shadow.json", "第七次進度報告素材/資料/door_coopen_core_bench_6_compare.json", evidence])
    B.table(p,["改善對象","本平台對照結果","證據層級"],[
        ["完整抽屜策略時間（前期）","平均 93.62 → 80.72 s","三對完整策略；非單一因素"],
        ["預測設定與運動品質（前期）","TCP p95：10.78–16.64 → 7.27–9.75 mm\n反轉率：26.01–28.42 → 1.12–2.33 次/s","N=1／N=5 三對；成本累加有差異"],
        ["模型預測一致性（本期）","一區間鉛直殘差 RMS\n0.776 → 0.045 mm","因果影子重播，非接觸閉迴路"],
        ["持握位置誤差（本期）","40–43.7° 平均 4.48 mm\n→ 1.33／1.32／1.35 mm","實測分箱三趟相近，非嚴格單變因"],
    ],widths=[8500,13500,9300],row_h=2300,fs=16)
    B.banner(p,"目前主張：執行介面一致的 W-GMPC 實作與實證；尚未證明全面優於外部方法。")

    p = S("接下來：由單一操作走向同趟導航—操作—返回", "05 / 研究目標與後續交付",
          "最後回到上次抽屜的基礎。近期先取得同一趟開門到內側換抓的接觸驗證與影片，再接通過中拉關和退出，最後串接既有抽屜開關及返回。導航段沿用 GMPC 加導航 CBF，不把導航的保護外推成手臂全身 CBF。成功子段要留影片，完整流程另外用同一趟連續紀錄驗證，不拼接不同版本。對外部文獻的同平台效能比較若要主張優越，還需另外設計；這是論文定稿證據缺口，不作為現在整趟整合的額外前置。", ["reviews/20261027_howard_confirmed_plan.md", evidence])
    for i,(a,b) in enumerate([
        ("開門 → 換抓", "先取得同趟實際閉爪與保持證據\n成功子段保留影片"),
        ("通過中關門", "在整機清空前已有實測關門進展\n再完成放手、退臂、交還導航"),
        ("開關抽屜 → 返回", "接回上次已建立的操作能力\n導航 GMPC ＋導航 CBF 實際啟用")]):
        B.card(p,a,b,1250+i*10600,4000,10100,5700,B.BLUE if i==0 else B.ORANGE)
    B.txt(p,"本期回答的是：如何把可規劃的動作，轉成位置伺服平台可執行的全身操作。",1450,11100,30000,2300,23,B.NAVY,True)
    B.txt(p,"已支持：限定配置的時間、預測一致性與持握誤差改善。\n待支持：完整物理通行、同趟往返及跨方法優越性。",1450,13800,30000,1850,18)
    B.banner(p,"完整流程仍是目標；依報告前取得的實際證據更新，不預寫成成功。")

    p = S("備用：最新換抓模型與實作狀態", "附錄 A / 與舊版完整拉關分開",
          "最新換抓名目模型承接實錄的實測關節、設定點、套用命令和偏移。新抓法沿內側橫桿軸轉三十度；參考與控制同時修訂後，走到內側保持，實際模型狀態的取樣 stop 和 warn 為零。收臂仍有一百二十毫米 TCP 落後及關節飽和，固定零點二九秒規劃壓力的自碰診斷未過。這不是力確認，也不包含通過拉關。整合進度以 r66 為快照，不以未完成的真節點聯測冒充實跑。", ["evaluation/results/door/door_regrasp_ref_v4_wgmpc_v2.json", "evaluation/results/door/door_regrasp_ref_v4.json", "reviews/20261028_152918_request.md"])
    B.table(p,["項目","結果","限制"],[
        ["控制狀態","承接 bench_6 的 q／s／u_prev／偏移","非同趟實際延續"],
        ["新內側抓法","φ＋30°；實測准入條件","名目模型無接觸力"],
        ["W-GMPC 名目追蹤","完成；取樣 stop／warn 0","收臂落後 120 mm、部分關節飽和"],
        ["整合與下一步","同一真節點＋替身做完整聯測","尚未取得新換抓 Isaac 實證"],
    ],widths=[7200,12100,12000],row_h=2200,fs=16)
    B.banner(p,"0.29 s 壓力核對自碰 4.90 mm，未過；本模型成功僅限所列名目條件。",color=B.ORANGE)

    p = S("備用：視覺預抓取與目標誤差", "附錄 B / 本期其他進展",
          "視覺支線先做到限定預抓取。相對鎖定目標到位誤差一點八三毫米，保持零點五一秒；但對真值的 TCP 誤差約三點九九毫米。這說明控制到位不能代替目標幾何正確。像素中心契約的候選版，在八十二格共同工具目標上，接近軸絕對誤差中位由一點三六降到零點九七毫米，但仍有變差影格，抓取放行誤差界尚未建立。這不是本期已做完視覺抓取或泛化的證據。", ["第七次進度報告素材/規格/DL3_run_result.yaml", "第七次進度報告素材/規格/PC2_result.yaml"])
    B.image_fit(p,MAT/"圖表/16_PC2_像素契約新版對照_誤差與餘裕.png",1250,3500,20000,11000)
    B.metric(p,"1.83 mm","相對鎖定目標的\n預抓取到位誤差",22000,3900,10000)
    B.metric(p,"1.36 → 0.97 mm","82 格共同工具目標\n接近軸絕對誤差中位",22000,8800,10000)
    B.banner(p,"預抓取到位 ≠ 可放行抓取；未閉爪、未操作，也未證明跨物件泛化。",color=B.ORANGE)

    p = S("代表性文獻與本研究定位", "附錄 C / 原始來源・2026/10/28 查核",
          "文獻表用來定位問題與方法，不是用不同平台報告的成功率作比較。主要三篇為 Wang 的全身優先序轉換、Sinaei 的耦合門通行規劃、Du 的混合介面雙層 MPC。Jang 用來確認門開關通行已有人做，Minniti 說明互動模型自適應不是新概念，Tang 則是導航 GMPC 的方法來源。Du 這裡比較二零二四年 v1，已知另有二零二六年期刊版本，不混用其數字。", [R1,R2,R3,R4,R5,R6,evidence])
    refs = [
        ("[R1] Wang, Chen, Zhao — ICRA 2025", "Whole-Body MPC for Mobile Manipulation with Task Priority Transition", R1),
        ("[R2] Sinaei et al. — arXiv v1, 2026", "Motion Planning for Mobile Manipulators Navigating Doorways via MPC", R2),
        ("[R3] Du et al. — arXiv v1, 2024", "An Efficient Representation of Whole-body MPC for Online Compliant Dual-arm Mobile Manipulation", R3),
        ("[R4] Jang, Kim, Park — RA-L 2023", "Motion Planning of Mobile Manipulator for Navigation Including Door Traversal", R4),
        ("[R5] Minniti et al. — ICRA 2021", "Model Predictive Robot-Environment Interaction Control for Mobile Manipulation Tasks", R5),
        ("[R6] Tang et al. — RA-L 2024", "GMPC: Geometric MPC for Wheeled Mobile Robot Trajectory Tracking", R6),
    ]
    for i,(a,b,url) in enumerate(refs):
        y = 3300+i*2000
        B.txt(p,a,1450,y,30000,700,18,B.BLUE,True)
        B.video_link(p,b,url,1450,y+800,30000)
    B.banner(p,"比較表的『方法不同』不是『已經比較勝出』；本平台改善數字另列。",color=B.GRAY)

    B.DOC.unlockControllers()
    for ext,filt,fn in [("pptx","Impress MS PowerPoint 2007 XML",B.DOC.storeAsURL),("pdf","impress_pdf_Export",B.DOC.storeToURL)]:
        fn(uno.systemPathToFileUrl(str(OUT/f"{STEM}.{ext}")),(B.U.pv("FilterName",filt),B.U.pv("Overwrite",True)))
    B.DOC.close(True)
    desktop.terminate()
    lines = ["# 第七次進度報告 v2｜從抽屜接續的逐頁講稿", "", "2026-10-28 整理；14 頁主簡報＋3 頁備用。正式日期待填。", "",
             "敘事：上次完成的抽屜 → 新問題 → 文獻差異 → 自己的方法 → 改善證據 → 全流程下一步。", "",
             "相關論文差異與本平台基線改善分列；未做外部方法同平台重現，不宣稱全面勝出。", ""]
    for row in B.SLIDES:
        lines += [f"## {row['n']:02d}｜{row['title'].replace(chr(10),' ')}","",row['notes'],"","來源：",""]
        for src in row['sources']:
            target = src if src.startswith('https://') else (ROOT/src).as_posix()
            lines.append(f"- [{src}]({target})")
        lines.append("")
    (OUT/"逐頁講稿與來源_v2.md").write_text("\n".join(lines),encoding="utf-8")
    (OUT/"投影片索引_v2.json").write_text(json.dumps(B.SLIDES,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Built v2: 14 main slides + 3 appendix.")


def preview_and_audit():
    preview = OUT/"預覽_v2"
    preview.mkdir(exist_ok=True)
    subprocess.run(["pdftoppm","-scale-to","1100","-png",str(OUT/f"{STEM}.pdf"),str(preview/"slide")],check=True)
    rows = json.loads((OUT/"投影片索引_v2.json").read_text(encoding="utf-8"))
    files = sorted(preview.glob("slide-*.png"))
    assert len(files)==len(rows)==17
    canvas = Image.new("RGB",(1320,6*283),"#e8edf3")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
    for i,file in enumerate(files):
        im=Image.open(file).convert("RGB"); im.thumbnail((420,237))
        x,y=10+(i%3)*440,32+(i//3)*283
        canvas.paste(im,(x,y)); draw.text((x,y-25),f"{i+1:02d}"+(" / appendix" if i>=14 else ""),font=font,fill="#17365d")
    canvas.save(OUT/"預覽總覽_v2.png")
    with zipfile.ZipFile(OUT/f"{STEM}.pptx") as z:
        for i in range(1,18):
            assert len(''.join(ET.fromstring(z.read(f'ppt/slides/slide{i}.xml')).itertext()))>70
        assert len([n for n in z.namelist() if n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml')])==17
        rel = z.read('ppt/slides/_rels/slide17.xml.rels').decode()
        assert all(url.replace('&','&amp;') in rel for url in (R1,R2,R3,R4,R5,R6)), 'Missing paper hyperlink'
    missing = sorted({s for row in rows for s in row['sources'] if not s.startswith('https://') and not (ROOT/s).is_file()})
    assert not missing, missing
    print('Checked: 17 rendered slides, text, speaker notes, literature hyperlinks, all local sources.')


if __name__ == '__main__':
    if '--preview-only' not in sys.argv:
        build()
    preview_and_audit()
