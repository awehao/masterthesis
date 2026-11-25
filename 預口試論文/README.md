# 預口試計畫書工作目錄

從中午第三章插圖開始的 Overleaf 手動修改，依 [完整修改清單](overleaf/Overleaf_從第三章插圖開始_修改清單.md) 逐項進行；[22 張第三章圖的複製格式](overleaf/第三章_現行22張圖_插入片段.tex) 已統一整理。

更新：2026-11-24。正文採用學姐 Tsai 的 XeLaTeX 格式，現行修改以 `overleaf/` 內分節 `.tex` 與唯一書目主檔為準。

目前住宅任務為：**開門通行 → 茶几圓柱移動中抓取 → 車尾收納盒 → 開抽屜取筆並收納 → 關抽屜 → 回程推門穿過再關門 → 回起點**。不同版本的完成情形與失敗仍依各趟紀錄分別報告。

| 內容 | 現行來源 |
|---|---|
| 中英文摘要 | [Markdown](00_摘要.md)、[中文 TeX](overleaf/abstract_zh.tex)、[英文 TeX](overleaf/abstract_en.tex) |
| 第一章 | [1.1–1.2](overleaf/第一章_前言與動機_含圖.tex)、[1.4](overleaf/第一章_本文架構.tex)、[1.3.5 比較表](overleaf/01_03_05_research_comparison.tex)；1.3.1–1.3.4 完整稿目前不在本資料夾 |
| 第二章 | [來源索引](02_研究系統介紹.md) |
| 第三章 | [來源索引](03_研究方法與演算法設計.md) |
| 第四章 | [實驗方法與結果分析](overleaf/04_ExperimentalResults.tex)、[本章引用文獻](overleaf/第四章_參考文獻_完整.bib) |
| 第五章 | [結論與未來工作](overleaf/05_Conclusions.tex) |
| 章節規劃 | [章節架構](論文章節架構.md)、[LaTeX 骨架](overleaf/chapters_outline.tex) |
| 參考文獻 | [thesis_references.bib](overleaf/thesis_references.bib)、[使用方式](參考文獻.md) |
| 本次修訂與待驗證項 | [修訂說明](overleaf/20261124_內容修訂說明.md) |
| 摘要上傳包 | [thesis_abstract_overleaf.zip](thesis_abstract_overleaf.zip)；入口 main.tex，目前僅編譯摘要 |
| 本次正文修訂包 | [20261124_計畫書修訂稿.zip](20261124_計畫書修訂稿.zip)；供替換學姐專案中的片段 |

`03_01` 至 `03_07` 為第三章唯一撰寫來源。兩份合併稿由 `python3 預口試論文/overleaf/build_manuscript_materials.py` 產生；修改分節後重新執行，不在合併稿另修一套內容。

關節展開與後段併線已有第 101 趟圓柱階段的 Isaac 證據，3.5.4 正文已更新；第 101 趟整體任務仍因回程門扇視覺估測錯誤而未完成。伸展評分正權重尚待實作與驗證，已完成批次對應權重為零。

2026-11-26：第四、第五章現行正文為上表所列 LaTeX；同名 Markdown 仍是較早工作稿。第五章依第四章已呈現的結果整理，不能用舊批次替新版本背書。舊 Markdown、分章書目及舊摘要預覽已留存 [歷史版本](歷史版本/)，本次整併前的完整文字備份為 `歷史版本/20261124_整併前備份.zip`。

本次未新增模擬實驗；本機未取得 XeLaTeX/BibTeX 編譯器，更新後的完整排版請在 Overleaf 檢查。舊 PDF 預覽不代表本次修訂內容。

英文參考方法圖已加入正文：[圖檔與插入格式](英文參考圖與Overleaf格式.zip)、[來源與放置位置](overleaf/figsrc/reference_figures/SOURCES.md)。

第一至三章已補比較表、平台與資料集表、八張圖及演算法框：[新增位置與使用方式](overleaf/第一至三章_新增內容與插圖位置.md)、[新增圖表與 Overleaf 內容包](第一至三章_新增圖表與Overleaf內容.zip)。
