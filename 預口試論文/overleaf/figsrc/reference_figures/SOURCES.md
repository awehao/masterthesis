# 英文參考方法圖

核對與擷取日期：2026-11-24。以下四張保留原圖英文標籤、顏色與幾何，只裁出圖面並另附中文圖說；圖說使用「引自」，不標為本研究自行繪製。原始論文、來源頁次、裁切範圍與 SHA-256 見 [manifest.json](manifest.json)。

| 本文位置 | 圖面 | 原始來源與圖號 | 引用鍵 |
|---|---|---|---|
| 2.4.1 | [Navigation2 模組分工](nav2_architecture_macenski2020.pdf) | Macenski 等人，The Marathon 2，Fig. 2；[作者 PDF](https://arxiv.org/pdf/2003.00368v2)，PDF 第 3 頁 | `macenski2020nav2` |
| 3.1.1 | [旋轉群與角度參數](rotation_representation_tang2024.pdf) | Tang 等人，GMPC，Fig. 1；[作者 PDF](https://arxiv.org/pdf/2403.07317v1)，PDF 第 1 頁；書目使用 RA-L 2024 期刊版 | `tang2024gmpc` |
| 3.4.1 | [卡爾曼預測／更新循環](kalman_predict_update_welch2004.pdf) | Welch、Bishop，TR 95-041，Figure 1-2；[原報告的大學保存本](https://www.cs.utexas.edu/~pstone/Courses/393Rfall15/readings/Welch+Bishop-TR-95.pdf)，PDF 第 6 頁 | `welch2004kalman` |
| 3.5.1 | [Mask R-CNN 原理](mask_rcnn_framework_he2017.pdf) | He 等人，ICCV 2017，Figure 1；[CVF 正式論文](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)，PDF 第 1 頁 | `he2017maskrcnn` |

## 與本研究的對應

- Mask R-CNN 圖說明候選區域、類別／外框及遮罩的關係。本文實作採 ResNet-50-FPN v2，模型訓練及遮罩深度估測另外交代，不將原圖當作本文逐層網路規格。
- 卡爾曼圖為一般線性濾波循環；圖中 A、H 對應本文 F_o、H_o，障礙模型的外部控制輸入取零。參考調整層的歷史位移估測與一階平滑另依正文處理。
- GMPC 圖只解釋旋轉表示的連續性，不使用原論文的差速車平台圖代替本文全向底盤。
- Nav2 圖是既有系統架構背景。本文採用的規劃及代價地圖功能，與自行設計的全身預測控制分別說明。

## 版本核對

Welch–Bishop 實際下載的 PDF 封面與頁尾均為 **March 1, 2004** 修訂版，因此新增 `welch2004kalman`；未把搜尋結果中的 2006 版日期套到這張圖。

Mask R-CNN 的 CVF 本文印刷頁碼與 IEEE DOI 書目頁碼有差異；圖面定位使用 PDF 第 1 頁的 Figure 1，既有 IEEE DOI 書目不因擷取圖面而改寫。

下載原文保留在 `originals/` 供追溯；交付 ZIP 僅包含選定圖面、插入片段及來源資料。英文標籤不改成標楷體；中文前導文字與圖說由論文範本排版。

## 使用方式

四張圖已加入各分節正文與自動合併稿。[英文參考圖_插入格式.tex](../../英文參考圖_插入格式.tex)供在學姐專案複製，已有相同圖與標籤時不重複插入。所有引用鍵統一在 [thesis_references.bib](../../thesis_references.bib)。

`extract_reference_figures.py` 只從既有 PDF 擷取圖面，未生成或修改原方法內容。Mask R-CNN 原圖含照片，其餘圖面亦依原 PDF 結構保留，不宣稱所有來源圖皆為純向量。
