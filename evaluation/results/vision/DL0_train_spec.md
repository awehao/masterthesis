# DL0 最小訓練規格（draft-1，送審；核可前不安裝環境、不訓練）

2026-10-05。依 Codex 子包結果審查（選 A）：單一 RGB 預訓練實例分割、固定切分／權重來源／種子／訓練上限／模型選擇規則；不做模型大搜查。
學習式遮罩（L1）與真值輔助遮罩（G1）共用**同一幾何後端**（`dl0_g1_check.g1_detect`：遮罩點 → D1 圓柱擬合 → D1 端點規則），不預設學習式較好。

## 1　模型與權重
- torchvision **Mask R-CNN ResNet-50-FPN v2**，權重 `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`（COCO 預訓練；下載後記 sha256）。
- 類別：背景＋`handle_bar`（1 類）；預測頭換成 2 類，其餘骨幹與 FPN 由預訓練權重初始化。
- 輸入：RGB 640×480（不縮放裁切，保持與內參一致）；不使用深度（RGB-D 比較不在本包）。

## 2　標籤與 ignore
- 標籤來源：`dl0_autolabel.py` 現行版本（含抽查修正：ignore 只取可見遮罩外側一圈）；**抽查摘要 `DL0_review_summary.yaml`**。
- 每格最多 1 個實例：可見遮罩非空 ⇒ 1 個實例（框＝可見遮罩外接框）；`no_handle_in_view` 或可見遮罩為空（例如全被近裁切）⇒ 負樣本（無實例）。
- **ignore 排除於損失**：torchvision 的遮罩損失不支援 ignore ⇒ 以自訂遮罩損失取代 `roi_heads.maskrcnn_loss`：對 RoI 投影後的 ignore 圖給權重 0，其餘像素正常計 BCE；
  評估時 ignore 像素不計入 IoU 分子與分母。框與分類損失不變。（以單元測試核對：ignore 全為 1 的 RoI 遮罩損失為 0；ignore 為 0 時與原損失相同。）

## 3　切分（以群組為單位，沿用 `DL0_data_spec.md` §5、§11）

| 切分 | 群組 | 影格 | 用途 |
|---|---|---|---|
| 訓練 | `static_v0`、`traj_c0b1_p3r_H5`、`traj_d1s4b_M`、`synth_dev_near` | 40＋518＋438＋49 | 微調 |
| 開發／驗證 | `traj_wg4b_f02_P`、`traj_mt_b1_02_P` | 87＋99 | 選 epoch、確認門檻 |
| 封存測試 | `dl0_test_lateral`、`dl0_test_view` | 82＋82 | 凍結後一次評估 |

- `traj_c0b1_p3r_H5` 是同一軌跡三次擷取（高度重複）⇒ **群組均衡取樣**：每個 epoch 各訓練群組的抽樣權重與群組大小成反比，避免單一軌跡主導。
- 所有衍生、增強與修標影格跟隨原群組。

## 4　訓練設定（固定，不搜尋）
- 種子 0（Python／NumPy／torch，cudnn deterministic）；batch 2；SGD lr 0.005、momentum 0.9、weight decay 1e-4；
  **上限 20 epoch**（每 epoch 600 次迭代，群組均衡取樣）；前 1 epoch 線性 warmup；第 15 epoch lr ×0.1。混合精度。
- 增強（僅訓練）：亮度／對比／飽和度抖動 ±0.2、水平翻轉 p = 0.5（橫桿左右對稱；翻轉只用於訓練，評估不用）。
- 熱與資源：CPU 熱中止沿用 92 °C 上限（以同類取樣器監看 `cpu_c`，超溫即停並保存最後檢查點）；資料載入 2 個 worker。

## 5　模型選擇與門檻（事前固定）
- 每 epoch 在**開發群組**評估：遮罩 IoU（ignore 排除；以分數最高的單一實例、分數門檻 0.5、遮罩二值化 0.5）。
- 選開發 IoU 最高的 epoch；同分取較早者。**分數門檻 0.5、二值化 0.5 固定**，不依開發結果調整。
- 凍結內容：權重 sha256、本規格、前處理、幾何後端（`dl0_g1_check.g1_detect` 與 D1 常數）、門檻。

## 6　開發評估（訓練後、凍結前）
在開發群組報：遮罩 IoU 與邊界、軸線（L1）有效率與角度、中心（L2）有效率與誤差（分距離箱）、拒絕原因、誤認與違反、每格推論耗時（GPU）。
G0（凍結 D1）、G1（真值輔助遮罩參考）、L1（學習式遮罩）三者同影格、同座標、同拒絕規則；G1／L1 都略過寬度篩選（無前板平面），G0 有，照實註明。

## 7　封存測試（凍結後一次）
凍結後才對 `dl0_test_lateral`／`dl0_test_view` 產生自動標籤並評估 G0、G1、L1；全部影格、失敗保留；不據測試結果調參或換路徑。
只支持「同一把手在新觀測條件下」的表現，不支持跨物件泛化。

## 8　環境
- 新建隔離 venv `~/venvs/dl0`（使用者層，不需 sudo）：torch／torchvision 的 CUDA 12.8 版（RTX 5060，8 GB），版本與 wheel 雜湊記錄。
- **不動** Isaac 的 venv（其內 torchvision 的 `nms` 運算子與 torch 版本不相容，已觀察到 `operator torchvision::nms does not exist`）。
- 預估：約 1,200 格、20 epoch × 600 迭代，GPU 約 30–60 分鐘。

## 9　待審
1. 模型（Mask R-CNN R50-FPN v2、COCO）與自訂 ignore 遮罩損失做法。
2. 群組均衡取樣、訓練上限與模型選擇規則（開發 IoU、固定門檻 0.5）。
3. 抽查由 Claude 目視初查（非人工）是否足夠作為本包前提，或需 Howard 複查後才訓練。
4. 建立 `~/venvs/dl0` 並下載 torch（約 3 GB）與 COCO 權重是否核可。
