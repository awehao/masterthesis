這個資料夾是依你最後貼出的第三章整理的替換版本，保留 3.1～3.7 與全部 27 個小節。

**在 Overleaf 只需要做三件事**

1. 用本資料夾的 03_ResearchMethodology.tex 完整替換原第三章內容。主檔仍維持原本的引入方式：

       \input{03_ResearchMethodology}

2. 將 figsrc/ch03 內的 13 張 PNG 上傳至 Overleaf 相同路徑。同名圖片需使用本包版本，尤其是橫排四格的 dynamic_obstacle_prediction.png。本文沒有使用 PDF 圖片。
3. 將「第三章_新增參考文獻.bib」的三筆書目貼到現有 thesis.bib。原有書目保留；若已有相同 key，合併成一筆即可。

本檔是完整第三章，不要同時引入原稿、舊版 3.1～3.4 或 3.4～3.7 合併稿，避免同一章與標籤出現兩次。

**本次刪減**

| 項目 | 原稿 | 新版 |
|---|---:|---:|
| 編號公式環境 | 98 | 37 |
| 圖片 | 21 | 13 |
| 主節 | 7 | 7 |
| 小節 | 27 | 27 |

公式統計以一個 equation 環境為一組；詳細原始碼與中文字符統計見「核對結果.json」。這些數字不是 PDF 頁數，實際頁數需以你的主檔、字級與行距編譯確認。

刪除基礎旋轉與雅可比的逐項展開、重複成本函數、QP Hessian 展開、一般濾波細節、候選評分的次要分項公式，以及以文字即可說明的任務與停止布林式。保留增廣模型、輪級限制、CBF、操作選位、關節展開與後段併線、機構約束及執行補償。

模型係數、部分進度與參數定義改為行內公式；沒有將長推導整段搬到附錄。

**圖片調整**

以下五張均未出現在新版正文或 PNG 上傳清單：

- visual_time_alignment.png
- tcp_target_compensation.png
- omni_wheel_velocity_feasible_set.png
- grasp_state_transition.png
- command_execution_timeline.png

另刪除 base_pose_error、joint_optimization、mask_rcnn_target_pipeline 與 model_bias_estimation，避免與保留圖片或文字重複。

三張原先引用文獻圖的地方，改用本研究自行繪製的圖：

| 位置 | 新圖 | 處理方式 |
|---|---|---|
| 3.1.1 | rotation_representation_redrawn.png | 重新畫連續旋轉與角度包絡跳變，引用 Tang 的幾何表示概念。 |
| 3.4.1 | kalman_tracking_redrawn.png | 以本章追蹤符號重畫預測、量測修正及回授。 |
| 3.5.1 | mask_rcnn_architecture.png | 依實際 ResNet-50-FPN v2 設定畫完整推論架構，取代原論文簡圖。 |
| 3.4.2 | dynamic_obstacle_prediction.png | 改為一排四格；後三格均表示同一未來時刻的不同候選。 |

Mask R-CNN 新圖包括 ResNet-50 殘差瓶頸、C2～C5 的區塊數、FPN、P2～P6、RPN、RoIAlign、偵測分支與遮罩分支。七類輸出包含背景，研究目標為六個前景類別。圖表示推論流程，遮罩分支承接偵測後的外框。

舊圖片與舊稿保留在原資料夾，本次只在替換包排除它們。四張新圖另附 SVG 圖源；Overleaf 正文只引用 PNG。

**方法描述的調整**

- 保留「抓持後物件與末端之間的相對位姿保持固定」作為閉鏈模型前提。
- 沒有使用手指位置、接觸力或連續證據公式作為本章研究方法。
- 沒有加入模擬器的物件連接實作說明；抓持辨識與滑移處理留作未來工作。
- 保留門與抽屜的機構模型、回程共轉關門、相對底盤持物及完整住宅任務。
- 保留模擬門角在機構進度與幾何核對中的資料用途，避免改寫為純視覺閉迴路。
- IoU 0.754、資料集統計與實驗參數留在第二章／第四章。本章只說明資料來源與方法。
- 3.7.4 保留英文演算法，精簡為背景求解與物理步執行的主要步驟。

**演算法設定**

沿用主檔的 algorithm、algpseudocode 與 setspace 設定，不需重複載入套件。若輸入、輸出仍顯示中文，可將原本兩行改成：

    \algrenewcommand{\algorithmicrequire}{\textbf{Input:}}
    \algrenewcommand{\algorithmicensure}{\textbf{Output:}}

章標題保留 c:methodology，並保留原稿 c:Mathodology 作為相容標籤。原主節與小節標籤保留。刪除公式的標籤另列於「刪除標籤清單.csv」；本章已沒有引用它們，若其他章引用過，需依內容改為小節引用或改寫敘述。

**完成的核對**

已檢查括號、環境、演算法區塊、重複標籤、交叉引用、書目 key、圖片路徑與圖說標籤位置，並確認全部圖檔為 PNG。新繪圖經文字邊界與方框容納檢查；保留圖片由目前 PDF 圖源重新匯出，並裁去上下多餘空白。保留圖片採接近全文寬度呈現，維持字級可讀性；不使用可能過時的 PNG 預覽。

本機未安裝 XeLaTeX，因此沒有宣稱已完成你實際 Overleaf 專案的整本編譯，也沒有估算確切 PDF 頁數。
