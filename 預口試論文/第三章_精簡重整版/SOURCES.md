新圖皆以本地繪圖程式自行繪製，並非將文獻原圖轉為黑白或裁切後使用。概念與模型來源於正文引用；圖說直接描述圖中內容。

**幾何表示**

rotation_representation_redrawn.png 以自行選定的角度範圍展示 SO(2) 的連續方向與角度主值跳變；版面、座標與資料曲線皆重新建立，並非實驗數據。

研究概念來源為 [Tang 等人的 GMPC 期刊論文](https://doi.org/10.1109/LRA.2024.3381088)；原稿使用的書目 key 為 tang2024gmpc。

**卡爾曼追蹤**

kalman_tracking_redrawn.png 以本章的 F_o、H_o 重畫預測、量測創新、修正與下一輪回授，使用標準線性卡爾曼模型。參考 [Welch 與 Bishop 的卡爾曼濾波介紹](https://www.cs.utexas.edu/~pstone/Courses/393Rfall15/readings/Welch%2BBishop-TR-95.pdf)，沿用 welch2004kalman；原始方法另引用 kalman1960filter。

**Mask R-CNN 完整架構**

mask_rcnn_architecture.png 核對本地 evaluation/house_seg.py 的 build_model 設定與以下一次來源：

- [Mask R-CNN 原論文](https://arxiv.org/abs/1703.06870)：兩階段偵測與實例遮罩方法。
- [ResNet 原論文與正式書目](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)：骨幹網路與殘差瓶頸。
- [FPN 原論文與正式書目](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)：側向連接與由上而下的多尺度特徵。
- [Torchvision Mask R-CNN 官方原始碼文件](https://docs.pytorch.org/vision/main/_modules/torchvision/models/detection/mask_rcnn.html)：v2 的雙層 RPN 卷積、四層卷積加單一全連接偵測頭，以及遮罩卷積頭。
- [Torchvision Faster R-CNN 官方原始碼文件](https://docs.pytorch.org/vision/main/_modules/torchvision/models/detection/faster_rcnn.html)：偵測 RoIAlign 尺寸與預測層。

本圖畫的是本研究組態的推論流程。輸入為三通道 480×640；ResNet 四階段區塊數為 3、4、6、3；P2～P5 特徵供 RoIAlign，P2～P6 供 RPN。偵測及遮罩預測層改為七類，包含背景。遮罩分支使用推論後處理的偵測框，最後選取對應類別並回復至影像尺寸。實際權重版本與資料來源仍以本研究實驗紀錄為準。

**動態障礙橫排圖**

dynamic_obstacle_prediction.png 為本研究候選運動概念圖，沒有引用外部圖片或實驗軌跡。(a) 是目前觀測；(b)～(d) 皆為同一未來時刻，分別比較原速度、減速與側移。機器人圖示為四輪十字全向配置。

**其餘保留圖片**

九張既有方法圖由預口試論文/overleaf/figsrc/ch03 的目前 PDF 圖源重新匯出 PNG，裁去上下多餘空白並保留邊界；沒有沿用可能過時的 PNG 預覽。其來源為本研究既有方法繪圖程式及 URDF 幾何，均屬方法示意。mobile_approach_distance_profile 為規則參數示例，不是某一趟實驗的量測速度或實際展開曲線。

本次新圖產生程式：

    預口試論文/overleaf/figsrc/ch03/build_condensed_figures.py

核對日期：2026-11-25。
