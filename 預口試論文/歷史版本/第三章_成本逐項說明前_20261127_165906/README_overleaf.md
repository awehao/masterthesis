# Overleaf 現行檔案與編輯方式

2026-11-27：第三章請使用 [現行精簡與一致性修訂稿](../第三章_精簡重整版/03_ResearchMethodology.tex)，含 48 組公式、13 張 PNG 及 7 節／27 小節；[替換方式](../第三章_精簡重整版/README_替換方式.md)。以下 22 張圖與分節合併稿說明屬較早版本，不要整份與現行稿同時載入。

2026-11-25 圖面修訂：[第三章圖片與圖說更新包](../第三章_圖片字級與圖說修訂.zip)。
19 張自繪圖已放大文字並移除圖內底部長句，分圖意義移至 LaTeX 圖說。
以約 14.7 cm 寬排版時，主要文字約 11 pt；正文建議使用 `0.98\textwidth`。
逐圖替換格式見 [第三章_圖說更新.tex](第三章_圖說更新.tex)。

**從中午第三章插圖開始逐項修改，請先看 [完整修改清單](Overleaf_從第三章插圖開始_修改清單.md)。**
圖說、標籤與確切位置整理於 [第三章 22 張圖插入片段](第三章_現行22張圖_插入片段.tex)。

更新：2026-11-24。格式沿用使用者提供的 `Tsai_thesis.zip`，Compiler 選 XeLaTeX。`main.tex` 載入 `thesis.tex`，目前預設仍只編譯摘要。

| 檔案 | 用途 |
|---|---|
| `abstract_zh.tex`、`abstract_en.tex` | 現行中英文摘要，含住宅整合任務 |
| `nthuvars.tex`、`nthuthesis.cls`、`preamble.tex` | 封面變數、學姐範本及字型／排版設定 |
| `第一章_前言與動機_含圖.tex`、`第一章_本文架構.tex` | 第一章 1.1–1.2、1.4；1.3.5 比較表另見 01_03_05_research_comparison.tex |
| `02_platform_and_interfaces.tex` | 2.1–2.2；平台規格、座標配置與命令鏈 |
| `02_software_and_integration.tex` | 2.3–2.4；含 Smac 2D、AMCL + EKF、Mask R-CNN 資料與示例 |
| `02_workflow_and_recording.tex` | 2.6–2.7；住宅任務與記錄方式 |
| `../第三章_精簡重整版/03_ResearchMethodology.tex` | 第三章現行來源，精簡文字並核對現行程式 |
| `03_01_kinematics.tex` 至 `03_07_execution_compensation.tex` | 較早第三章分節來源，現行稿不由這些檔案重建 |
| `第三章_3.1至3.4完整稿.tex`、`第三章_3.4至3.7完整稿.tex` | 較早自動整併稿；含重疊 3.4，不與現行稿同時載入 |
| `04_ExperimentalResults.tex` | 第四章 4.1–4.6；包含各項比較、住宅整合及方法限制 |
| `05_Conclusions.tex` | 第五章；研究成果與貢獻、未來研究方向，含成果彙整表 |
| `第四章_參考文獻_完整.bib` | 第四章實際引用的五筆文獻，均已收錄於唯一書目主檔 |
| `chapters_outline.tex` | 標題骨架；不要與完整正文同時載入 |
| `thesis_references.bib` | 全論文唯一書目主檔 |
| `第三章_圖說更新.tex` | 舊版 22 張圖的 figure 替換段落；現行 13 張圖依精簡稿處理 |
| `第一至三章_新增內容與插圖位置.md` | 本次八張新增圖、四份表格與演算法框的使用位置及證據 |
| `algorithm_preamble.tex` | 演算法框導言設定；本地 preamble 已載入 |

在學姐專案上修改時，上傳所需分節及 `figsrc` 圖片，再替換對應內容。書目統一使用：

```latex
\bibliographystyle{IEEEtran}
\bibliography{thesis_references}
```

若原 `thesis.bib` 有學姐專案其他章節仍需要的條目，先將獨有條目併入主檔，再切換；不要重複定義相同引用鍵。引用顯示問號時，確認 `\cite{key}` 與主檔鍵名一致，再於 Overleaf 使用 Recompile from scratch。

`python3 預口試論文/overleaf/build_manuscript_materials.py` 只更新較早分節合併稿及目錄，不會更新現行 `第三章_精簡重整版/03_ResearchMethodology.tex`。

2.1、2.2 與 1.3.5 已補入；2.5 完整 LaTeX 及第一章 1.3.1–1.3.4 未在本資料夾收錄。本資料夾尚不是完整五章的可直接編譯論文。現行第三章僅引用精簡稿內的 PNG，舊 PDF 圖源不代表本次正文的用圖清單。

本次檢查範圍與證據見 [20261124_內容修訂說明.md](20261124_內容修訂說明.md)。舊分章書目及舊 Markdown 已留存上層歷史版本。本機無 XeLaTeX/BibTeX，本次未產生新版摘要或整份論文 PDF。

2026-11-26：第四章與第五章現行正文分別為 `04_ExperimentalResults.tex` 與 `05_Conclusions.tex`；第五章承接第四章結果，不沿用較早 Markdown 中尚未完成住宅任務的敘述。第五章以章節及圖表交叉引用整理本研究成果，未新增外部文獻。第四章分章書目供核對及合併使用，請依引用鍵去重；完整專案仍統一使用 `thesis_references.bib`。

另補四張英文參考方法圖：Nav2（2.4.1）、旋轉表示（3.1.1）、卡爾曼濾波（3.4.1）、Mask R-CNN（3.5.1）。[插入片段](英文參考圖_插入格式.tex)及[來源清單](figsrc/reference_figures/SOURCES.md)均已整理；分節稿已含圖，不重複貼入。

較早新增圖表及重製來源見 [第一至三章_新增內容與插圖位置.md](第一至三章_新增內容與插圖位置.md)。該版第三章為 22 張圖，現行精簡稿為 13 張；不要將舊「16 張」或「22 張」插圖包整份併入現行稿。遮罩例圖執行既有權重的離線推論，未啟動模擬或訓練。
