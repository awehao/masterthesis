# 第一章新增圖片來源

整理日期：2026-11-21。四個圖檔均直接取自官方或論文作者來源，保留原始檔案，不裁切、重畫、翻譯或移除原圖內容。LaTeX 以排版方式組合平台照片，中文分圖標籤由論文字型顯示。

## 工業與室內導向之平台實例

建議放在 1.1.1「自主移動機械臂之發展與應用」，接在介紹移動底盤與機械手臂組成的段落後。兩張照片合為一個 figure。

### kuka_kmr_iisy_application.jpg

- 來源：[KUKA KMR iisy 官方產品頁](https://www.kuka.com/en-de/products/amr-autonomous-mobile-robotics/mobile-robot-systems/kmr-iisy-autonomous-mobile-cobot)。
- [原始照片](https://www.kuka.com/-/media/kuka-corporate/images/products/mobility/kmr-iisy-mobile-cobot/autonomous-mobile-cobot--kmr-iisy.jpg?hash=AF7408DB11279112124F5A067B3A35E9&rev=c32c518312fa49fcabc04c8874417e50&w=1900)。
- 原圖 1900 × 802；產品與操作人員之官方照片，用於說明平台形式，非本研究實機。
- 引用鍵：`kukakmriisy`。

### stretch4_platform_views.jpg

- 來源：[Hello Robot Stretch 4 官方產品頁](https://hello-robot.com/stretch-4/)。
- [原始照片](https://hello-robot.com/wp-content/uploads/2026/05/quad_green-e1778479775445.jpg)。
- 原圖 1000 × 600；呈現同一款 Stretch 4 的不同機構配置姿態，不是四款不同機器人，亦非自主任務成功紀錄。
- 引用鍵：`hellorobotstretch4`。

## mittal2022_articulated_fig1.png

- 建議位置：1.1.2「動態環境下之導航與操作需求」。
- 來源：[Mittal 等人論文 v2，Fig. 1](https://arxiv.org/html/2103.10534v2)。
- [原始圖檔](https://arxiv.org/html/2103.10534v2/teaser_iros3.png)。
- [作者專案](https://www.pair.toronto.edu/articulated-mm/)。
- 正式發表：M. Mittal, D. Hoeller, F. Farshidian, M. Hutter, and A. Garg, “Articulated Object Interaction in Unknown Scenes with Whole-Body Mobile Manipulation,” in *2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 1647–1654, 2022, doi: 10.1109/IROS47612.2022.9981779。
- 原圖 2036 × 1822。包含輪式機器人的模擬場景、關節式物件與足式平台實機畫面；圖說明確標示含模擬與實機，不將全部畫面都稱為輪式實機成果。
- 引用鍵：`mittal2022articulated`。

## burgess2023_motm_banner.png

- 建議位置：獨立插圖版放在 1.3.2「全身模型預測控制與移動中操作」；依使用者正文整理的「第一章_前言與動機_含圖.tex」則改放在 1.2.1，承接該小節首次介紹 MotM 的段落。同一章不重複放置此圖。
- 來源：[Manipulation On-The-Move 作者專案](https://benburgesslimerick.github.io/ManipulationOnTheMove/)。
- [原始圖檔](https://benburgesslimerick.github.io/ManipulationOnTheMove/images/MotmBanner.png)。
- 原圖 3840 × 1032，為作者專案中的模擬連續動作示意，不是實機連拍，也不是本研究結果。
- 方法之正式發表：B. Burgess-Limerick, C. Lehnert, J. Leitner, and P. Corke, “An Architecture for Reactive Mobile Manipulation On-The-Move,” in *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 1623–1629, 2023, doi: 10.1109/ICRA48891.2023.10161021。
- 引用鍵：方法用 `burgess2023motm`；專案圖片來源用 `burgessmotmproject`。

## 引用與檔案

- 正文與圖說片段：`overleaf/第一章_新增插圖片段.tex`。
- BibTeX：`overleaf/第一二章_新增圖片參考文獻.bib`；只將尚未存在的引用鍵加入學姐專案的 `thesis.bib`，不新增第二套 bibliography 指令。
- 檔案大小、SHA-256 及原始網址記錄於同資料夾的 `supplementary_images_metadata.json`。
- 原有兩張 IFR 圖的來源仍見 `SOURCES.md`。
