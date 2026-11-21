# 第二章黑白架構圖

兩張圖均為黑線白底，文字使用學姐模板內的標楷體（DFKai-SB，`fonts/kaiu.ttf`）。PDF 為向量圖且已嵌入所用字型，上傳 Overleaf 後不需另行安裝。請保留下列路徑：

- `figsrc/ch02/system_control_architecture.pdf`：整體控制架構設計，放在 2.4 節開頭。
- `figsrc/ch02/navigation_architecture.pdf`：底盤導航架構，放在 2.4.1。

將 `第二章_插圖片段.tex` 中的兩段內容分別複製至上述位置。`第二章_架構圖插入.tex` 提供相同片段，兩份擇一使用，避免重複圖號標籤。片段沿用學姐的 `figure`、`includegraphics`、`caption`、`label` 與 `cref` 寫法，不需修改主檔或額外加入繪圖套件。

PNG 供預覽；SVG 與 `draw_architecture.py` 供修改原稿。重新執行繪圖程式時，需保留本專案原有的 `fonts/kaiu.ttf`，或以 `THESIS_KAIU_FONT` 指定相同字型檔的位置。導航圖沿用第五次進度報告第 4 頁的三路輸入布局，並更新命令處理鏈；整體控制圖呈現本研究的系統整合設計。來源與圖面適用範圍見 `figsrc/ch02/SOURCES_architecture.md`。

已確認兩張 PDF 可正常渲染、中文字型已嵌入、圖中文字未超出方框，以及插入片段的圖片路徑與標籤一致。未在本機編譯完整學姐論文專案。
