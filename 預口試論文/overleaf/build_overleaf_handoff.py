"""Generate one handoff guide and copyable figure snippets from current section sources."""
from pathlib import Path
import re,json

ROOT=Path(__file__).resolve().parent
FIGURES={
 'rotation_representation_tang2024':('旋轉群與角度參數比較','3.1.1 末尾、3.1.2 之前；說明為何採用旋轉群表示之後'),
 'base_pose_error':('底盤參考與實際位姿','3.1.2 幾何誤差及局部模型的公式與說明之後'),
 'wholebody_motion_coupling':('底盤側移與手臂補償','3.1.3 全身正向運動學、雅可比矩陣說明之後'),
 'arm_setpoint_tracking':('設定點與實際關節追蹤','3.2.2 增廣狀態及設定點／實際位置差異的說明之後'),
 'omni_wheel_velocity_feasible_set':('十字全向輪速度可行域','3.3.2 輪速限制、可行域推導及超限例子之後'),
 'joint_optimization':('底盤與手臂共同求解','3.3.3 QP 建構與數值求解說明之後'),
 'kalman_predict_update_welch2004':('卡爾曼預測與量測更新','3.4.1 等速度模型與量測模型的公式之後；保留本文符號對照段'),
 'dynamic_obstacle_prediction':('原速度、減速與側移預測','3.4.2 最小預測間距的公式及候選說明之後'),
 'navigation_command_layers':('導航與命令處理分層','3.4.5 原始雷射防護與輪級命令處理說明之後'),
 'mask_rcnn_framework_he2017':('Mask R-CNN 原論文架構','3.5.1 六類模型與訓練資料介紹之後、各物件深度估測之前'),
 'mask_rcnn_target_pipeline':('本研究視覺目標生成流程','3.5.1 深度反投影及各物件幾何估測說明之後、3.5.2 之前'),
 'visual_time_alignment':('影像與機器人狀態時間對齊','3.5.2 帶時間戳的座標轉換與有效性說明之後'),
 'visual_operation_reference':('操作目標與底盤選位','3.5.3 候選集合、評分式與選位機制之後'),
 'mobile_approach_distance_profile':('移動中接近的距離剖面','3.5.4 關節展開、後段併線及距離限速說明之後；3.6.1 只引用此圖'),
 'grasp_state_transition':('接近、閉爪與抓持確認','3.6.1 抓持證據與理想連接確認說明之後'),
 'door_operation_sequence':('開門前後的全身構型','3.6.2 旋轉門模型及門角更新公式之後'),
 'drawer_operation_sequence':('抽屜拉出前後的全身構型','3.6.2 平移抽屜模型及行程限制公式之後'),
 'base_relative_carrying':('相對底盤的持物運送','3.6.3 底盤相對持物目標公式與流程說明之後'),
 'command_execution_timeline':('命令在途、接續與停止時序','3.7.1 在途命令前推及生效初始狀態說明之後'),
 'model_bias_estimation':('模型偏置估測','3.7.2 偏置觀測、平滑及限幅公式之後，TCP 補償介紹之前'),
 'tcp_target_compensation':('TCP 有界目標補償','3.7.2 TCP 積分補償公式及完成條件說明之後'),
 'wholebody_control_workflow':('全身預測控制完整流程','3.7.4 第一段之後、演算法框之前'),
}

def main():
    rows=[];snippets=['% !TeX program = xelatex',
        '% 自中午第三章圖片修訂起，現行 22 張圖之完整插入片段。',
        '% 由 build_overleaf_handoff.py 從唯一分節正文產生；不在本檔另改圖說。',
        '% 分節正文已含圖：整節替換時，不再重複貼入本檔。',
        '% 本檔供逐段複製，不是可整份 input 的額外章節。',
        '% 圖號由 LaTeX 自動產生，勿手寫 3-1、3-2 等數字。','']
    for source in sorted(ROOT.glob('03_0[1-7]_*.tex')):
        content=source.read_text();section=int(source.name[3:5]);sub=0
        for m in re.finditer(r'\\subsection\{([^}]+)\}|\\begin\{figure\}.*?\\end\{figure\}',content,re.S):
            if m.group(1):sub+=1;continue
            block=m.group();path=re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',block).group(1)
            stem=Path(path).stem;title,place=FIGURES[stem];number=f'3.{section}.{sub}'
            before=content[:m.start()].rstrip();lead=before.rsplit('\n\n',1)[-1]
            lead='\n'.join(ln for ln in lead.splitlines() if not ln.lstrip().startswith('%')).strip()
            lead=re.split(r'\\end\{[^}]+\}',lead)[-1].strip()
            assert '\\label{' not in lead and '\\begin{' not in lead,stem
            label=re.search(r'\\label\{([^}]+)\}',block).group(1)
            rows.append(dict(section=number,title=title,path=path,placement=place,source=source.name,label=label))
            snippets+=['% ==================================================',f'% {number} {title}',
                f'% 位置：{place}',f'% 正文來源：{source.name}',
                '% 同名 PDF 請覆蓋上傳；同名 label 只保留一份。',
                '% ==================================================',lead,block,'','']
    assert len(rows)==22 and {Path(x['path']).stem for x in rows}==set(FIGURES)
    (ROOT/'第三章_現行22張圖_插入片段.tex').write_text('\n'.join(snippets))
    (ROOT/'overleaf_handoff_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    table='\n'.join(f"| {x['section']} | {x['title']} | [{Path(x['path']).name}]({x['path']}) | {x['placement']} |" for x in rows)
    doc=r'''# 這裡開始：從中午加入第三章圖片起的 Overleaf 修改清單

整理日期：2026-11-24。範圍涵蓋中午加入第三章示意圖、圖面重製、住宅任務與視覺修訂、英文參考圖，以及最後新增的第一至三章表格／圖表。以下是目前版本的一次性修改清單；你自行在學姐 Overleaf 專案操作。

本地內容、圖檔及引用結構已整理完成。完整頁面、圖表浮動與頁數尚需在你的 Overleaf 專案以 XeLaTeX 編譯確認。

## 1. 先選擇修改方式

**第三章建議依 3.1 → 3.7 整節替換。**目前每份 `03_0x_*.tex` 都已包含最新公式、正文、圖說與 `figure`，這樣能一起處理舊公式及缺少的標籤。若你想逐張插圖，改用 [22 張圖完整插入片段](第三章_現行22張圖_插入片段.tex)，依每段 `% 位置` 複製。

整節替換完成後，該節的圖片已在正文內，不再另外貼一次插圖片段。同一圖的 `\label` 只留一份。

## 2. 上傳圖片與設定

1. 將本次包內 `figsrc/ch03/`、`figsrc/reference_figures/` 及用到的 `figsrc/ch02/` 圖片上傳，維持原目錄。同名 PDF 以新版覆蓋，原檔名不變不代表圖面未更新。
2. Overleaf 的 `\includegraphics` 使用 PDF；PNG 作預覽，SVG 留供圖面編修。
3. 將 [algorithm_preamble.tex](algorithm_preamble.tex) 放入專案，在主檔 `\begin{document}` 前、`cleveref` 載入後加一次：

```latex
\input{algorithm_preamble}
```

演算法另需 `algorithm` 與 `algpseudocode`，設定檔已載入；正文使用的數學與表格套件為 `amsmath`、`amssymb`、`graphicx`、`array`、`setspace`，依學姐原專案已有套件保留。

## 3. 第三章：正文要改什麼

| 位置 | 改法 | 要複製的完整來源 |
|---|---|---|
| 第三章標題與開頭、3.1 | 統一章名「研究方法與全身模型預測控制器設計」；保留 GMPC、李群、運動學與座標定義；補三張圖 | [03_01_kinematics.tex](03_01_kinematics.tex) |
| 3.2 | 完整放入位置設定點與實際關節追蹤、增廣狀態、離散模型及凝縮預測式；補設定點追蹤圖 | [03_02_augmented_prediction.tex](03_02_augmented_prediction.tex) |
| 3.3 | 保留共同 QP；輪速限制後補十字全向輪可行域推導、超限例子與圖；末尾保留共同求解圖 | [03_03_objective_constraints.tex](03_03_objective_constraints.tex) |
| 3.4 | 以完整五小節替換：運動估測 → 未來衝突 → 速度／側移候選 → CBF → 雷射防護及命令處理；補三張圖 | [03_04_dynamic_obstacle_avoidance.tex](03_04_dynamic_obstacle_avoidance.tex) |
| 3.5.1–3.5.2 | 補六類 Mask R-CNN、模擬訓練、遮罩深度點、門把 ROI、帶時間戳的目標轉換；保留原論文圖並補本研究流程圖 | [03_05_visual_task_reference.tex](03_05_visual_task_reference.tex) |
| 3.5.3 | 選位評分加入水平伸展項及定義；保留「既有批次權重為零、正權重尚待實作／驗證」；替換舊選位圖 | 同上，搜尋 `% 3.5.3` |
| 3.5.4 | 整個小節替換舊卡氏接近式：距離進度、關節構型內插、預測底盤上的 FK、後段併線、距離限速圖 | 同上，搜尋 `% 3.5.4` |
| 3.6.1 | 補抓持證據、實機／模擬資訊差別；接近剖面引用 3.5.4 的圖；保留抓持狀態轉移圖 | [03_06_mobile_constrained_operation.tex](03_06_mobile_constrained_operation.tex) |
| 3.6.2 | 門與抽屜分別說明、分別放圖；移除舊合併操作圖 | 同上，搜尋 `% 3.6.2` |
| 3.6.3 | 持物運送後補車尾收納、後續取筆與關抽屜／關門銜接；保留底盤相對持物圖 | 同上，搜尋 `% 3.6.3` |
| 3.7.1–3.7.3 | 整合在途命令、生效時刻、實際放行輸入、偏置／TCP 補償、停止尾段與晚到處置 | [03_07_execution_compensation.tex](03_07_execution_compensation.tex) |
| 3.7.4 | 流程圖保留；把舊 `enumerate` 七點列表替換為目前 `algorithm` 框，說明非阻塞求解與持續執行 | 同上，搜尋 `% 3.7.4` |

`03_01_kinematics.tex` 已含第三章的 `\chapter`，請連同原章標題與開頭一起替換。其餘六份由 `\section` 起，不再加 `\chapter`。

**第三章由七份分節檔依序組成。**兩份「3.1 至 3.4 完整稿」與「3.4 至 3.7 完整稿」重疊於 3.4，勿整份同時貼入。若使用 `\chapters_outline.tex` 查看章名，僅作對照；正式目錄由正文標題自動產生。

### 3.2 的公式標籤

先前 3.3 引用不到的凝縮預測式，已在完整 3.2 內補齊：

```latex
\label{eq:ch3_condensed_augmented_prediction}
```

因此要一起更新 3.2，不能只貼 3.3 的新圖。完整來源保留各公式定義與交叉引用，不手動改公式號。

## 4. 第三章：22 張圖逐張放置表

以下列的是小節編號，**不是固定圖號**。實際「圖 3.x」由你的 Overleaf 依出現順序產生。每張的完整前導文字、圖說與標籤均在 [插入片段](第三章_現行22張圖_插入片段.tex)。

| 小節 | 圖片內容 | 上傳檔案 | 放在哪段後面 |
|---|---|---|---|
__FIGURE_TABLE__

## 5. 舊圖及舊說法的處理

| Overleaf 中的舊內容 | 這次的處理 |
|---|---|
| `articulated_operation_constraints.pdf` 合併開門／抽屜圖 | 移除原 `figure` 與對應 `\cref`；3.6.2 分成 `door_operation_sequence.pdf`、`drawer_operation_sequence.pdf` |
| `fig:ch3_articulated_operation_constraints` | 不再沿用此舊標籤；新圖標籤分別為 `fig:ch3_door_operation_sequence`、`fig:ch3_drawer_operation_sequence` |
| 舊 `visual_operation_reference.svg` 或舊 PDF | 覆蓋新版 `visual_operation_reference.pdf`，用 3.5.3 最新圖說；SVG 供編修 |
| 舊 `arm_setpoint_tracking.pdf` | 覆蓋新版 PDF；圖說用「手臂設定點與實際關節位置之追蹤關係」，保留設定點與實際位置的區分 |
| 差動／麥克納姆式底盤圖樣，或固定轉速下畫成菱形平移可行域 | 換目前四輪十字配置圖；菱形是 **vx–omega** 剖面，固定 omega 的 **vx–vy** 可行域為方框 |
| 舊 3.5.4 卡氏位姿混接式 | 整節換成關節展開與後段併線；舊稿已存歷史版本，不與新稿同時出現 |
| 「新展開法尚無 Isaac 執行證據」 | 改為第 101 趟圓柱階段已有分支、抓取、抬升及放盒證據；同時保留整趟在回程門扇視覺錯誤後停止的事實 |
| 「以持物返回為終點，暫不包含關門」 | 刪除，換成第 6 節的完整住宅任務 |
| 「抽屜僅作為比較案例」 | 改為住宅任務包含開抽屜、取筆與關抽屜；獨立比較仍另行報告 |
| 舊 3.7.4 七點條列 | 改為演算法框；上方控制流程圖保留 |

## 6. 摘要、第一章與第二章要同步的內容

完整任務統一為：

**開門通行 → 茶几圓柱移動中抓取 → 圓柱放車尾盒 → 開抽屜取筆並放盒 → 關抽屜 → 回程推門通行與關門 → 回起點。**

| 位置 | 要改什麼 | 圖片／表格及複製來源 |
|---|---|---|
| 中文、英文摘要 | 補住宅整合任務，仍以方法為主，不加入單趟成果數字 | [abstract_zh.tex](abstract_zh.tex)、[abstract_en.tex](abstract_en.tex)；沿用學姐摘要環境，不把另一份完整摘要章再貼一次 |
| 第一章 1.3.5 | 增加 GMPC、Burgess MotM、Mittal、Minniti、Du、Sun 比較與研究缺口 | [01_03_05_research_comparison.tex](01_03_05_research_comparison.tex)，只有此小節；1.3.1–1.3.4 保留你原稿 |
| 2.1 | 補輪徑 0.10 m、L=0.245 m、Lite 6 六軸／440 mm、夾爪、光達及手腕相機規格表 | [02_platform_and_interfaces.tex](02_platform_and_interfaces.tex)；區分官方硬體規格與模擬設定 |
| 2.1.2 | 加整機座標：平面基準、手臂基座、TCP、相機、光達 | [robot_coordinate_frames.pdf](figsrc/ch02/robot_coordinate_frames.pdf)，放座標配置介紹段落後 |
| 2.2.3 | 明寫底盤及手臂命令鏈；PD 600/60 是 Isaac 設定 | [command_execution_chains.pdf](figsrc/ch02/command_execution_chains.pdf)，放兩條命令介面說明後 |
| 2.4.1 | 明寫 Smac Planner 2D、AMCL 觀測與 EKF 融合分工；各趟控制位姿來源依實際配置分列 | [02_software_and_integration.tex](02_software_and_integration.tex) 中對應小節；可放 [Nav2 英文參考圖](figsrc/reference_figures/nav2_architecture_macenski2020.pdf) 於既有 Nav2 模組分工介紹後 |
| 2.4.2 | 補六類 Mask R-CNN、Isaac 語意標註訓練、遮罩選深度點與門把 ROI；明寫模擬訓練限制 | 同上小節；六類例圖 [mask_rcnn_class_examples.pdf](figsrc/ch02/mask_rcnn_class_examples.pdf)，接分割與深度介紹後，再放資料集表及 IoU 表 |
| 2.6 | 以完整住宅任務替換舊流程，加入切換條件與時間配置 | [02_workflow_and_recording.tex](02_workflow_and_recording.tex) 的 2.6；[任務狀態機](figsrc/ch02/residential_task_state_machine.pdf) 在開頭任務總述後，[第 97 趟甘特圖](figsrc/ch02/run97_task_gantt.pdf) 接狀態機及時間說明後 |
| 2.7 | 保留資料記錄、時間戳與重現方式，與最新執行記錄欄位一致 | 同上檔案的 2.7；沒有額外新增圖 |

### 第二章已放好的舊圖，請保留

本次 2.1–2.2 與 2.4 的文字檔未包含你在 Overleaf 另外貼入的全部舊圖。若整節替換文字，先保留這些 `figure`，再放回下列位置：

| 舊圖／圖組 | 保留位置 | 原插入片段 |
|---|---|---|
| `mobile_manipulator_model.png` | 2.1 平台總述後；與新增座標圖用途不同 | [第二章_新增照片插入.tex](第二章_新增照片插入.tex) |
| `wrist_camera_gripper_model.png` | 2.2.1 手腕相機安裝介紹後 | 同上 |
| `simulation_environment_map.pdf` | 2.3.2 場景建置段落後 | [第二章_模擬環境地圖插入.tex](第二章_模擬環境地圖插入.tex) |
| `isaac_door_area.png`、`isaac_table_area.png` 圖組 | 2.3.2 地圖／場景配置介紹後 | [第二章_新增照片插入.tex](第二章_新增照片插入.tex) |
| `system_control_architecture.pdf` | 2.4 節開頭的整體資訊流介紹後 | [第二章_架構圖插入.tex](第二章_架構圖插入.tex) |
| `navigation_architecture.pdf` | 2.4.1 本研究導航與障礙資訊分工說明後；與原論文 Nav2 架構分開 | 同上 |
| `door_handle_rgb.png`、`door_handle_depth.pdf` 圖組 | 2.4.2 RGB-D 量測介紹後；六類分割示例接在後續 Mask R-CNN 介紹後 | [第二章_新增照片插入.tex](第二章_新增照片插入.tex) |

## 7. 參考文獻怎麼改

先找學姐主檔中的 `\bibliography{...}`，確認實際正在讀取的 `.bib`。

- 若已使用 `thesis_references.bib`：上傳目前的 [書目主檔](thesis_references.bib)，保留你遠端另有且仍使用的獨有條目。
- 若仍使用學姐原本的 `.bib`：把所需完整條目合併到該檔，正文照目前的 `\cite{key}`；同一鍵只保留一筆。
- 參考圖與模型所需的鍵：`he2017maskrcnn`、`macenski2020nav2`、`fox2003kld`、`ufactorylite6specs`、`welch2004kalman`。
- 第三章其餘鍵也須保留：`tang2024gmpc`、`sola2018micro`、`grandia2019frequency`、`goldfarb1983dual`、`stellato2020osqp`、`kalman1960filter`、`ames2019cbf`、`mittal2022articulated`、`dijkstra1959note`、`flash1985coordination`、`burgess2023motm`。
- 比較表另使用 `minniti2019wholebody`、`du2026wholebody`、`sun2026motion`，並在討論中使用 `burgess2024reactive`；沿用完整期刊／會議書目，不把 preprint 標題混入已發表期刊條目。

目前書目共 32 個唯一鍵。來源在 `.bib` 內已用 `%` 分組；文獻編號與目錄、圖號都由 LaTeX 自動產生。不要把 `SOURCES.md` 當作 `.bib` 貼入。

## 8. 這幾個數值與說法請照目前稿

| 項目 | 寫法 |
|---|---|
| 手臂基座位置 | 車心前 0.11 m；相對 `base_link` 高 0.28 m，相對平面 `{B}` 高 0.33 m（含 0.05 m 固定高度） |
| 六類資料集 | 訓練 4432、驗證 166、測試 595 張；同類可見畫面可能重疊，不把逐類計數相加為總張數 |
| IoU | 表中逐類是驗證 IoU，六類平均 0.8012；測試摘要平均 0.754 分開寫；逐類測試數據尚未留存，不能用驗證成績代填 |
| 第 97 趟甘特圖 | 使用 Isaac 時間；流程結束 387.70 s、最終模擬記錄 388.71 s；抽屜仍有 11.1 mm 殘留開度 |
| 新展開法 | 第 101 趟圓柱階段完成；整趟因回程門扇視覺估測錯誤而停止 |
| 距離剖面圖 | 是逐軸速度上限與距離展開規則的設計曲線；實際進度另受時間及狀態限制 |
| 正權重伸展項 | 尚待實作與驗證；已完成對照批次採權重零 |
| 夾持成功 | 模擬用手指狀態／接觸證據，理想連接用於抓持後操作；不把這些訊號寫成 Lite 6 實機夾爪原生回授 |

## 9. 修改完的最後核對

- 正式目錄來自目前正文的 `\chapter`、`\section`、`\subsection`；不要另載入只有標題的 `\chapters_outline.tex`。
- 搜尋舊 `fig:ch3_articulated_operation_constraints`、舊卡氏 3.5.4、`暫不包含關門` 及 `抽屜僅作比較`，依上述位置替換。
- 22 張第三章圖各有一個 `figure`，開門及抽屜是兩張獨立圖；3.6.1 只引用接近剖面圖，不重放。
- 若公式或引用仍顯示 `??`／`[?]`，先確認對應 `\label`／BibTeX 鍵已載入，再使用 Overleaf 的 Recompile from scratch。
- 本地已核對提供稿的標籤、引用鍵及圖檔；Overleaf 原稿其他內容需由實際編譯結果一併確認。
'''
    (ROOT/'Overleaf_從第三章插圖開始_修改清單.md').write_text(doc.replace('__FIGURE_TABLE__',table))
    print('Generated one complete guide and 22 current figure snippets.')

if __name__=='__main__':main()
