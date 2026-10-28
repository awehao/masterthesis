#!/usr/bin/env python3
"""Expand previous drawer mechanisms and add an explicit progress review.

Loads the checked v2 deck, replaces five pages, and changes their presentation
order. All historical evidence retains its original scope. No simulation runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import uno
from PIL import Image, ImageDraw, ImageFont

import build_slides as B
from build_slides_v2 import STEM as PREV_STEM

ROOT, OUT = B.ROOT, B.OUT
SIX = ROOT/"第六次進度報告素材"
STEM = "第七次進度報告_進度回顧與抽屜流程改善_v3"
ORDER = [1,2,3,5,6,13,4,7,8,9,10,11,12,14,15,16,17]
ROWS = []


def reset_page(old_number, title, section, notes, sources):
    p = B.DOC.getDrawPages().getByIndex(old_number-1)
    while p.getCount():
        p.remove(p.getByIndex(0))
    B.txt(p,section,1250,470,28000,500,11,B.BLUE,True)
    B.txt(p,title,1250,1140,31300,1500,28,B.NAVY,True)
    B.U.add_line(p,B.DOC,1250,2800,B.W-1250,2800,B.LINE,30)
    B.txt(p,"第七次進度報告｜工作稿 v3｜2026/10/28 整理・報告日期待填",1250,18000,29000,550,9,B.GRAY)
    B.txt(p,str(old_number),B.W-2400,18000,1000,550,11,B.GRAY,align=2)
    ROWS[old_number-1].update(title=title,section=section,notes=notes,sources=sources)
    np=p.getNotesPage()
    body=next((np.getByIndex(i) for i in range(np.getCount()) if np.getByIndex(i).getShapeType()=="com.sun.star.presentation.NotesShape"),None)
    if body is not None:
        body.String=notes+"\n\n來源：\n"+"\n".join(sources)
    return p


def build():
    global ROWS
    ROWS=json.loads((OUT/"投影片索引_v2.json").read_text(encoding="utf-8"))
    desktop=B.U.connect_office()
    B.DOC=desktop.loadComponentFromURL(uno.systemPathToFileUrl(str(OUT/f"{PREV_STEM}.pptx")),"_blank",0,())
    B.DOC.lockControllers()
    scope="上次已完成的機制，這次補充修改原因與效果；不當作新的重跑批次。"
    specs="evaluation/results/wgmpc_drawer_open_close.yaml"
    task="evaluation/drawer_task_node.py"
    sources=["第六次講稿.md","第六次進度報告素材/簡報用實驗比較表.md"]

    p=reset_page(2,"進度回顧：導航與完整抽屜操作已建立","01 / 進度回顧",
        "先回顧整體進度。第一階段建立 GMPC 導航與預測式 CBF；第二階段建立底盤三維加手臂六維的全身模型，再將設定點追蹤納入 W-GMPC。上次已展示導航、實際夾持、抽屜開啟二百毫米、關閉、釋放和返回。這次會把上次一頁流程中沒有詳細說明的修改展開，回答啟動怎麼穩、底盤和手臂怎麼分工，以及退出收臂如何接順。導航曾有 CBF 的成果，不等於前期抽屜影片所有相位都已啟用該模組。",sources)
    B.table(p,["階段","已建立的能力","本次承接"],[
        ["第一階段：移動","GMPC 路徑追蹤、動態障礙預測與導航 CBF","前往操作區與返回的基礎"],
        ["第二階段：全身控制","底盤 3 ＋手臂 6 聯合求解\n設定點追蹤模型與命令限制","讓命令、預測與實際回授一致"],
        ["上次：完整抽屜操作","夾持 → 開啟 200 mm → 關閉\n放開 → 退開 → 收臂 → 返回","已有完整流程，不只單一末端到位"],
        ["這次報告","展開流程平順化的修改與實測\n再介紹鉸鏈門與接觸切換","從做得到，進一步說明怎麼做順"],
    ],widths=[7500,13500,10300],row_h=2200,fs=16)
    B.banner(p,"本次順序：進度回顧 → 抽屜流程修改 → 文獻與方法 → 開門延伸 → 後續工作。")

    p=reset_page(3,"抽屜流程怎麼接順？先看三個容易卡住的地方","02 / 完整流程與修改位置",
        "上次完整抽屜流程分成九個階段。為了讓整趟流暢，我把問題分成三個位置：第一是導航交給操作、求解器剛上線時，命令與狀態延遲不能積壓。第二是接近與開關期間，底盤引導、名目手臂姿態和末端追蹤要一致，不能讓底盤和手臂互相拉扯。第三是放手後的退出收臂，目標不能一次跳太遠，回程計畫也不能阻塞主迴圈。流暢不是把所有保持時間刪掉，必要的夾持確認、保持和安全停止仍保留。",[task,"evaluation/dual_source_executor.py",specs,"第六次講稿.md"])
    phases=["導航接近","低速接手","展開對準","閉爪確認","協同開啟","關閉保持","放手退開","同步收臂","交還返回"]
    for i,label in enumerate(phases):
        x=1250+(i%3)*10600; y=3450+(i//3)*2350
        B.rect(p,x,y,10100,1850,B.PALE,B.LINE)
        B.txt(p,f"{i+1}  {label}",x+250,y+450,9600,1000,21,B.BLUE if i<6 else B.ORANGE,True)
        if i%3<2:
            B.txt(p,"→",x+10100,y+500,500,800,17,B.BLUE,True)
    B.txt(p,"① 接手／啟動：避免舊訊息積壓，承接最後套用命令。\n② 接近／開關：底盤與手臂共同分擔，維持把手相對關係。\n③ 退開／返回：目標斜坡、同步收臂、回程計畫不阻塞。",1450,11100,30000,3850,20)
    B.banner(p,"流暢＝減少不必要的擺盪、繞路與停發；不取消必要的到位、夾持與保持確認。")

    p=reset_page(5,"修改一：清除命令積壓，讓延遲補償對上實際","02 / 啟動與接近段的平順化",
        "啟動時的擺盪不能只靠調權重解釋。抽屜模擬端原本每個物理步只處理一則 ROS 訊息，多個話題會排隊，命令實際生效比求解器假設晚很多。相關診斷中，狀態到生效的延遲最大到三百九十毫秒，而補償假設約八十毫秒。後來改成每物理步有界清空待處理回呼，命令只取最新一筆。兩趟修正版最大延遲約九十毫秒，啟動高度峰對峰為一點四與一點五毫米。這是抽屜這條命令鏈的結果，不代表另一條全身模擬器的所有高頻振盪都已解決。",[specs,"第六次進度報告素材/圖表/07_啟動擺盪_TCP高度.png","第六次進度報告素材/產生程式/build_sixth_progress_motm.py","第六次進度報告素材/資料/motm_190426/motm_metrics.json"])
    B.card(p,"原本：每物理步只處理一則訊息","多話題積壓 → 命令較晚生效\n補償假設約 80 ms，實際最大 390 ms",1250,3500,15000,3400,B.GRAY)
    B.card(p,"修改：有界清空回呼，命令取最新","每步處理上限 64，不無限等待\n兩趟修正版：最大約 90 ms",17300,3500,15000,3400,B.BLUE)
    B.image_fit(p,SIX/"圖表/07_啟動擺盪_TCP高度.png",1250,7450,31300,6500)
    B.txt(p,"修正版兩趟：啟動 TCP 高度峰對峰 1.4／1.5 mm。",1450,14300,30000,1150,22,B.GREEN,True)
    B.banner(p,"既有抽屜證據，本次展開說明；圖中多次修改為探索過程，非嚴格單變因實驗。",color=B.GRAY)

    p=reset_page(6,"修改二：退開、收臂與回程不要重新卡一下","02 / 退出與控制交還的平順化",
        "這頁把最後幾個不必要動作具體說明。第一，放手後不直接把 TCP 目標跳到六十毫米外，而是讓退開量以斜坡增加，避免手臂瞬間衝到速率上限。第二，收臂原本讓 j3 最後動，肩部先立起來，夾爪會先升高約四十四公分再落下。改成整個速率向量同比例縮放，讓各軸同時到達；j3 的下界保護仍保留。這一對紀錄的收臂時間由十一點七五秒到六點四八秒，修正版高度約五百四十七至五百六十三毫米。第三，收臂時提早非阻塞發布回程計畫，讓交還導航不必因主迴圈停發而停住。這些是不同階段的修正，不能把所有節省時間直接相加。",[task,"evaluation/dual_source_executor.py",specs,"第六次進度報告素材/圖表/08_收臂夾爪高度_修正前後.png"])
    B.image_fit(p,SIX/"圖表/08_收臂夾爪高度_修正前後.png",1250,3350,19800,10500)
    B.metric(p,"11.75 → 6.48 s","該對紀錄的收臂時間\n不是整個任務時間",21800,3700,10700,B.GREEN)
    B.card(p,"放手 → 退開","退開目標改斜坡\n避免一次跳 60 mm",21800,8400,10700,2900,B.BLUE)
    B.card(p,"收臂 → 回程","提早非阻塞發布回程計畫\n交還時承接最後套用命令",21800,11700,10700,3100,B.ORANGE)
    B.banner(p,"同步收臂保留 j3 下界保護；不同階段的局部結果，不拼成新的整趟時間改善數字。",color=B.GRAY)

    p=reset_page(13,"抽屜流程的改善：效率與運動品質分開看","02 / 流程平順化的實驗證據",
        "前面解釋的是怎麼把流程接順，這一頁再用正式對照看效果。策略比較中，整機協調和嚴格停車各三趟都通過，完整任務平均八十點七二對九十三點六二秒，逐對縮短十三點三八到十四點零五百分比。另一批比較同一核心的一步與五步，五步整控制窗的 TCP 位置誤差 p95 和手臂命令反轉率較低。這兩張表來自不同批次，統計窗口不同。命令反轉率不是物理振動量測；三對也不支持普遍優越性。這次重報時，要明確說這是上期已完成的數據，本期用來把修改機制說清楚，而非新增重跑。",["第六次進度報告素材/簡報用實驗比較表.md","evaluation/results/motm_speed/park_hold_formal_summary.yaml","evaluation/results/horizon_ablation/c0b1_results.yaml"])
    B.txt(p,"完整任務策略：整機協調 vs 嚴格停車（各 3 趟）",1450,3450,30000,1000,21,B.NAVY,True)
    B.table(p,["指標","嚴格停車","整機協調"],[
        ["平均完整任務時間","93.62 s","80.72 s"],
        ["任務判準","3/3 通過","3/3 通過"],
    ],x=1250,y=4700,widths=[11000,10150,10150],row_h=1500,fs=18)
    B.txt(p,"同一核心的預測設定：N=1 vs N=5（另一批，各 3 趟）",1450,10000,30000,950,21,B.NAVY,True)
    B.table(p,["整控制窗指標","單步 N=1","五步 N=5"],[
        ["TCP 位置誤差 p95","10.78–16.64 mm","7.27–9.75 mm"],
        ["手臂命令反轉率","26.01–28.42 次/s","1.12–2.33 次/s"],
    ],x=1250,y=11200,widths=[11000,10150,10150],row_h=1500,fs=17)
    B.banner(p,"上表為完整策略比較；下表 N 改變時成本累加也改變。不同批次，不混算時間或歸因。",color=B.GRAY)

    # Update cover, chapter labels, and footer page numbers without changing old data.
    cover=B.DOC.getDrawPages().getByIndex(0)
    for i in range(cover.getCount()):
        sh=cover.getByIndex(i)
        st=getattr(sh,'String','')
        if st.startswith("由抽屜操作到門通行"):
            sh.String="由抽屜流程平順化到門通行：\nW-GMPC 方法改善與流程整合"
        elif st.startswith("上次：協同開關抽屜"):
            sh.String="回顧：完整抽屜操作\n本次：展開平順化修改，再延伸到門操作"
    ROWS[0]['title']="由抽屜流程平順化到門通行：W-GMPC 方法改善與流程整合"
    ROWS[0]['notes']="本次先回顧上次已完成的完整抽屜流程，再展開啟動、退開、收臂與控制交還的具體修改和數據。接著比較文獻方法，介紹開門模型更新及接觸成果；上次數據可重報，但不改稱本期新增實驗。"
    np=cover.getNotesPage()
    for i in range(np.getCount()):
        sh=np.getByIndex(i)
        if sh.getShapeType()=="com.sun.star.presentation.NotesShape":
            sh.String=ROWS[0]['notes']
    mapping={old:new for new,old in enumerate(ORDER,1)}
    for old in range(1,18):
        p=B.DOC.getDrawPages().getByIndex(old-1)
        for j in range(p.getCount()):
            sh=p.getByIndex(j)
            st=getattr(sh,'String','')
            if st.startswith("第七次進度報告｜工作稿"):
                sh.String=st.replace("v2","v3")
            elif st==str(old) and sh.Position.Y>=17800:
                sh.String=str(mapping[old])
    B.DOC.unlockControllers()
    target=OUT/f"{STEM}.pptx"
    B.DOC.storeAsURL(uno.systemPathToFileUrl(str(target)),(B.U.pv("FilterName","Impress MS PowerPoint 2007 XML"),B.U.pv("Overwrite",True)))
    B.DOC.close(True)
    # OOXML's presentation list defines slide order; slide/notes/media relations stay intact.
    with zipfile.ZipFile(target) as z:
        entries=[(info,z.read(info.filename)) for info in z.infolist()]
    patched=[]
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
    for info,data in entries:
        if info.filename=='ppt/presentation.xml':
            root=ET.fromstring(data); ids=root.find('p:sldIdLst',ns); original=list(ids)
            for child in original:
                ids.remove(child)
            for old in ORDER:
                ids.append(original[old-1])
            data=ET.tostring(root,encoding='utf-8',xml_declaration=True)
        patched.append((info,data))
    tmp=target.with_suffix('.reorder.tmp')
    with zipfile.ZipFile(tmp,'w') as z:
        for info,data in patched:
            z.writestr(info,data)
    os.replace(tmp,target)
    doc=desktop.loadComponentFromURL(uno.systemPathToFileUrl(str(target)),"_blank",0,())
    doc.storeToURL(uno.systemPathToFileUrl(str(OUT/f"{STEM}.pdf")),(B.U.pv("FilterName","impress_pdf_Export"),B.U.pv("Overwrite",True)))
    doc.close(True); desktop.terminate()
    final=[]
    for n,old in enumerate(ORDER,1):
        row=dict(ROWS[old-1]); row['n']=n; final.append(row)
    (OUT/"投影片索引_v3.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=["# 第七次進度報告 v3｜進度回顧與抽屜流程平順化", "", "2026-10-28 整理。14 頁主簡報＋3 頁附錄。上期結果可重報，但不更改批次或實驗時序。", "", "前半段：進度回顧 → 抽屜流程 → 啟動延遲 → 退開／同步收臂／回程 → 效果對照。", ""]
    for row in final:
        lines += [f"## {row['n']:02d}｜{row['title'].replace(chr(10),' ')}","",row['notes'],"","來源：",""]
        for src in row['sources']:
            url=src if src.startswith('https://') else (ROOT/src).as_posix()
            lines.append(f"- [{src}]({url})")
        lines.append("")
    (OUT/"逐頁講稿與來源_v3.md").write_text("\n".join(lines),encoding='utf-8')
    print('Built v3: 14 main slides + 3 appendix. Drawer mechanisms expanded; no new experimental claims.')


def preview_and_audit():
    dest=OUT/'預覽_v3'; dest.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm','-scale-to','1100','-png',str(OUT/f'{STEM}.pdf'),str(dest/'slide')],check=True)
    files=sorted(dest.glob('slide-*.png')); assert len(files)==17
    canvas=Image.new('RGB',(1320,6*283),'#e8edf3'); draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for i,path in enumerate(files):
        im=Image.open(path).convert('RGB'); im.thumbnail((420,237)); x,y=10+i%3*440,32+i//3*283
        canvas.paste(im,(x,y)); draw.text((x,y-25),f'{i+1:02d}'+(' / appendix' if i>=14 else ''),font=font,fill='#17365d')
    canvas.save(OUT/'預覽總覽_v3.png')
    rows=json.loads((OUT/'投影片索引_v3.json').read_text(encoding='utf-8'))
    missing={s for row in rows for s in row['sources'] if not s.startswith('https://') and not (ROOT/s).is_file()}
    assert not missing, missing
    text=subprocess.check_output(['pdftotext','-f','2','-l','6',str(OUT/f'{STEM}.pdf'),'-']).decode('utf-8')
    assert all(x in text for x in ['進度回顧','延遲','6.48','80.72'])
    print('Checked: 17 pages, revised order, drawer figures and source paths. Speaker notes preserved.')


if __name__=='__main__':
    if '--preview-only' not in sys.argv:
        build()
    preview_and_audit()
