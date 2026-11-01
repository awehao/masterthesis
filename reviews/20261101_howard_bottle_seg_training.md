> 2026-11-02 Howard 最新直接指示：不要停車抓取驗證，先試讓手腕相機在移動接近中看見更多瓶身與支撐區的姿態。優先交接見 [20261102_howard_moving_gaze_no_parked_grasp.md](20261102_howard_moving_gaze_no_parked_grasp.md)；暫不實跑上一輪 OBSERVE／parked 抓取分支，必要安全停止保留。以下是模型訓練歷史紀錄。

# Howard 已確認：先訓練瓶子辨識／分割，不訓練抓取控制策略

2026-11-01。Howard 明確選擇「先辨識／分割瓶子等物品」。Codex 接續離線訓練第一版 bottle-only RGB instance segmentation，不更換現有 GMPC／估計器，不啟動 Isaac。

## 本次範圍

- 模型沿用本機既有 torchvision Mask R-CNN ResNet50 FPN v2／COCO 初始化與 DL0 的 ignore-mask 損失；分類頭改成背景＋bottle。不是把舊把手權重當成瓶子模型。
- 使用現有底盤與手腕 RGB-D 紀錄。捕獲時刻真值＋圓柱幾何與深度只用於離線自動標註；模型輸入只有 RGB。標註不是人工真值，先抽查訓練遮罩。
- nominal（w1_1、compact_nominal1）只進 train；minus5 clean1 進 dev；plus5 w1_2 留作不參與 epoch 選擇的 test。兩台相機與衍生資料隨場景分組，去除完全重複 RGB。不把此同環境偏移測試稱為跨物體／跨高度盲測。
- 固定一次短程微調：seed 0、batch 2、6 epoch ×80 step、SGD lr .0025、前40 step warmup、第5 epoch起lr×.1；dev 等權分組 IoU 選較早最佳 epoch，門檻均 .5。凍結權重/hash後 test 一次，不依結果調參。
- 產物獨立存 `evaluation/dl0_ckpt/bottle_seg_v1_data/`、`evaluation/dl0_ckpt/bottle_seg_v1_train/`（沿用既有忽略大型權重規則），不覆寫 DL0 把手模型。

## 資源協調

- 訓練程式和 Isaac／重播共用 `masterthesis-isaac-<uid>.lock`；訓練中不能另外起 Isaac 或重播。已有直接啟動的 Isaac 程序也拒絕並行，不殺對方程序。
- CPU ≥92°C、GPU ≥88°C、熱量測失效即停，保存最後 checkpoint；結束後釋放鎖。非完成狀態不得聲稱訓練完成。
- Claude 不同時另訓練同一模型；可接續不搶 GPU 的接口設計，但不要將未驗證權重直接放進抓取閉迴路。

## 尚未解決的問題

本次資料仍是同一圓柱瓶、同材質、桌高0.60 m。第一版只是學習式遮罩起點，不能據此宣稱任意物品／高度泛化，不能用分割模型取代桌面三維估計、目標生成、IK與碰撞檢查。下一階段需要不同高度／背景／遮擋與外觀的分組資料，並以 mask＋實測深度生成三維目標；目前控制器的桌面高度先驗仍未移除。

## 本輪完成紀錄

- 已完成 480 步微調。首段在 step 240 收到外部 SIGTERM（來源未定位，非熱中止），保留檢查點後分段續至 320／400／480。續訓恢復權重、optimizer、scaler，不宣稱與不中斷訓練逐位元相同。step 240 的 dev 評估未完成；只在已完成的 dev 評估中選最佳權重，未用 test 選模。
- 最終模型：`evaluation/dl0_ckpt/bottle_seg_v1_continue3/best.pth`；SHA256 `e5077fb8ede8815f02230748b7b35746d104f363ac8c28ed04d0d6da1d2bc962`。初始化是 COCO，不是把手微調 checkpoint。
- 資料：train 383（227 正／156 負）、dev 191（112／79）、held-out 189（113／76）；微小／不可靠標註排除 5。訓練遮罩抽查包含遠近兩相機與負例，不是全資料人工逐格標註。
- dev 等權相機群組 IoU 0.9314，選 iteration 480。凍結後 test：底盤正例平均 IoU 0.8811、手腕 0.9746，等權平均 0.9278，負例誤檢 0／76。IoU 不是抓取成功率。
- 離線單張推論中位 22.66 ms，只是此 GPU／資料下的推論計時，不是 ROS 加控制整輪的實時性證據。
- 推論入口 `evaluation/bottle_seg_infer.py`，只收 RGB 圖片及權重，不收深度、桌高或瓶位真值；保存預測遮罩與 overlay。既有控制器、深度圓擬合估計器與抓取流程沒有替換。
- 小型可追溯摘要：`evaluation/results/vision/bottle_seg_v1_summary.json`。大型資料、權重及逐段訓練紀錄保留原目錄，不覆寫把手模型。

可直接離線試一張 RGB：

```bash
.venv-dl0/bin/python evaluation/bottle_seg_infer.py \
  --weights evaluation/dl0_ckpt/bottle_seg_v1_continue3/best.pth \
  --image INPUT_RGB.png --out NEW_OUTPUT_DIRECTORY
```

下一包是以學習式 mask 接實測深度的三維目標估計，先旁路核對不同高度與位置，再談控制採用。現有桌高先驗未移除，不能把此模型當成已完成任意高度抓取。是否已由 Claude 讀取仍待其回報。
