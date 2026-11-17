# Overleaf：中英文摘要與關鍵字

內容版本：v0.4，2026-11-17。題目沿用「動態環境下移動機械臂預測導航與受約束操作」。本包先排中文摘要及英文摘要，各頁附對應關鍵字，供後續套用系所論文範本。

## 上傳與編譯

1. 在 Overleaf 選擇 **New Project → Upload Project**，上傳 `thesis_abstract_overleaf.zip`。
2. 將專案的 **Compiler** 設為 **XeLaTeX**。
3. 確認 **Main document** 為 **main.tex**，按 **Recompile**。

ZIP 的根目錄直接包含 `main.tex`。Overleaf 的編譯器設定仍請明確選擇 XeLaTeX；TeX 編輯器魔法註解不等於修改網站設定。[上傳專案](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project)、[選擇編譯器](https://docs.overleaf.com/getting-started/recompiling-your-project/selecting-a-tex-live-version-and-latex-compiler)

## 檔案分工

| 檔案 | 用途 |
|---|---|
| `main.tex` | 可獨立編譯的主檔，載入兩份摘要 |
| `preamble.tex` | 字型、頁面、行距與關鍵字排版 |
| `abstract_zh.tex` | 中文摘要與關鍵字 |
| `abstract_en.tex` | 英文摘要與關鍵字 |
| `latexmkrc` | 本機 latexmk 的 XeLaTeX 設定 |

繁體中文使用 **Noto Serif CJK TC**，中文無襯線使用 **Noto Sans CJK TC**，英文使用 **TeX Gyre Termes**。中文依 ctex 排版，目標引擎為 XeLaTeX。[Overleaf 中文排版說明](https://www.overleaf.com/learn/latex/Chinese)、[Noto CJK 字型範例](https://www.overleaf.com/articles/demonstration-of-noto-serif-cjk-and-noto-sans-cjk-fonts/sgrwgcddtqsq)

## 套入既有論文範本

已有學校主檔時，可搬入 `abstract_zh.tex`、`abstract_en.tex` 的內容，並沿用學校的摘要標題及頁碼設定。關鍵字使用的 `\thesiskeywords` 定義在 `preamble.tex`；若範本已有關鍵字命令，改用範本提供的命令即可。

目前版面為 A4、12 pt、1.3 倍行距、左邊界 30 mm、其餘邊界 25 mm；這些是工作稿設定，正式格式依系所範本調整。摘要內容依現有導航、抽屜、單瓶及理想抓持門案例整理，完整住宅任務仍列待驗證；詳細來源保留在本機 Markdown 工作稿的證據索引中。

本次 LaTeX 與 `00_摘要.md` 的內容一致。之後在 Overleaf 修改摘要分檔時，請一併同步本機工作稿。

## 編譯檢查

2026-11-17 使用 Tectonic 0.17.0（XeTeX 核心）完成本機編譯，產生兩頁 A4 PDF：中文摘要及英文摘要各一頁，含各自關鍵字；編譯記錄未出現缺字或文字溢出版面的警告。已檢查輸出文字及兩頁版面。尚未登入 Overleaf 實際編譯；上傳時請依前述步驟選擇 XeLaTeX。
