# Overleaf：Tsai 版中英文摘要與關鍵字

版本：v0.7（格式沿用 v0.5），2026-11-17。**以使用者提供的 `Tsai_thesis.zip` 為主要格式來源**，`NTHU_PME_thesis_main.zip` 僅作差異比對。摘要更新為 `00_摘要.md` v0.7，以「本研究旨在建立一套」開頭，聚焦導航、執行動態建模與協同操作方法。最後僅交代模擬評估範圍，成果數據與比較由本機結果章呈現。

## 上傳與編譯

1. 在 Overleaf 選擇 **New Project → Upload Project**，上傳工作目錄的 `thesis_abstract_overleaf.zip`。
2. 將 **Compiler** 設為 **XeLaTeX**，**Main document** 設為 **main.tex**。
3. 按 **Recompile**。預設只輸出中英文摘要與關鍵字。

ZIP 根目錄含 `main.tex`，由它載入 `thesis.tex`。TeX 編輯器魔法註解不會自動修改 Overleaf 的編譯器設定。[上傳專案](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project)、[選擇編譯器](https://docs.overleaf.com/getting-started/recompiling-your-project/selecting-a-tex-live-version-and-latex-compiler)

## 檔案分工

| 檔案 | 用途 |
|---|---|
| `main.tex` | Overleaf 入口，載入 `thesis.tex` |
| `thesis.tex` | 使用 Tsai 的 `nthuthesis` 類別及摘要載入流程 |
| `nthuthesis.cls` | 從 Tsai 包原樣取用，保留授權文字及新版封面定義 |
| `preamble.tex` | 字型、標題、行距、摘要環境修正與關鍵字排版 |
| `nthuvars.tex` | 目前論文題目與待填封面資料 |
| `00_abstract.tex` | 摘要入口，依序載入中英文分檔 |
| `abstract_zh.tex` | 中文摘要及 `\keywordzh` 關鍵字 |
| `abstract_en.tex` | 英文摘要及 `\keyworden` 關鍵字 |
| `fonts/kaiu.ttf`、`fonts/times.ttf` | Tsai 包附的標楷體與 Times New Roman |
| `latexmkrc` | latexmk 的 XeLaTeX 設定 |
| `template/格式套用紀錄.md` | 範本比對、沿用項目及必要修正 |
| `template/README_Tsai_original.md` | 原範本說明與來源資訊 |

## 本次沿用的 Tsai 格式

- A4、`extbook` 14 pt；左邊界 3.17 cm，其餘邊界 1 inch。
- 中文標楷體、英文 Times New Roman；字型由專案內檔案載入。
- 中文摘要沿用 `\linespread{2.0}`；英文摘要使用 `spacing{1.5}`。
- 沿用章節標題格式、`\raggedbottom`、前置單面排版與小寫羅馬頁碼，中文摘要從 i 開始。
- 沿用 `abstractzh`、`abstracten`、`keywordzh`、`keyworden` 介面；關鍵字續行對齊內容。

這是對使用者提供範本的套用，正式送審仍依系所要求核對。原版浮水印預設關閉，本專案沿用此設定。

## 封面與正文後續擴充

`nthuvars.tex` 已填目前中英文題目；姓名、學號、組別、指導教授與口試年月保留明確待填欄位。`thesis.tex` 預設 `\includecoverfalse`，補齊資料後可改為 `\includecovertrue`，使用 Tsai 新版封面。

目前只載入摘要。第一至第五章仍以本機 Markdown 分章檔為撰寫來源，之後逐章轉為 LaTeX 並加入 `thesis.tex`。原範本的學長姓名、論文內容、實驗圖與參考文獻不載入此專案。

中英文摘要文字與本機 `00_摘要.md` 同步；之後在 Overleaf 修改摘要分檔時，也請同步本機工作稿。

## 編譯檢查

2026-11-17 以 Tectonic 0.17.0（XeTeX 核心）完成本機編譯，輸出兩頁 A4 PDF：中文摘要與英文摘要各一頁，含各自關鍵字及 i、ii 頁碼。已檢查兩頁版面、輸出文字與字型嵌入；編譯記錄未出現缺字、Overfull 或 Underfull 警告。

本機檢查沒有登入 Overleaf；上傳後仍須選擇 XeLaTeX。封面預設關閉，本次檢查範圍為摘要頁。

## 前言圖文稿（2026-11-21）

新增 `01_background.tex`，將產業需求與 AMMR 背景整合成單一「前言」節，配合兩張 IFR 官方原圖。文字對應上層 `1.1_前言_圖文稿.md`；圖檔位於 `figsrc/ch01/`，來源見同資料夾 `SOURCES.md`；本節引用集中於 `background_references.bib`。

可將獨立交付的 `前言_Overleaf圖文稿.zip` 上傳至 Overleaf，Compiler 選 XeLaTeX、Main document 選 `main.tex`；包內入口會載入 `background_preview.tex`，輸出第一章前言與參考文獻。若直接使用本資料夾，將 Main document 改選 `background_preview.tex` 即可。原 `main.tex` 仍為摘要入口。

合併到完整論文時，在第一章中 `\input{01_background}`，並將本節書目合併到全論文使用的 `.bib`。本輪已檢查引用鍵、圖檔路徑、原圖尺寸與來源，尚未於本機編譯此新稿。
