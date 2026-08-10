# 續作備忘 — 2026-08-09 深夜

## 狀態

無批次執行中,改動已由 auto-commit 提交(`11e26a14`)。可直接關機。

## 今天的結論(已驗證,不需重跑)

**根因鏈** — 為什麼碰撞降不到 0:

1. 感知看得見 ≠ 進入 CBF。cluster → track → age≥3 → 瞬時速度≥0.10 → 淨位移≥0.05,
   任一關擋掉,該物體就**完全沒有約束**。
2. seed27 的 track 全程存在(age 中位 218)、KF 位置準確,但只有 **23%** 的週期被發布。
3. 68% 的拒絕來自**瞬時速度閘門**:`dyn_obs_5` 設定速度就是 0.10,而門檻也是 0.10。
4. 降門檻到 0.05 只讓發布率 23%→32%、碰撞 9/9→7/10 —— **因為估計本身有偏**:
   KF 估 0.049 對真實 0.100,1.6 m 長箱的 cluster 質心沿可見面滑動。
   淨位移閘門吃同一個估計(0.098 m 對門檻 0.10 m),同樣卡在邊界。
5. 因此**任何建立在質心速度上的門檻都不可靠** → 需要不依賴分類的一層。

**raw-scan safety shield**(`scan_safety_shield.py`,插在 velocity_smoother 之後):

| 測試 | 結果 |
|---|---|
| seed27 十次重播 | 9/9 → **0/10**,p < 10⁻⁴ |
| 27 條配對路線 | 5 → **0**,修好 5 條、引入 0 條,McNemar p = 0.0625 |
| 到達率 | 27/27,未變成卡死 |
| 速度代價 | −0.1%,介入僅 1.1% 的週期 |
| 最差間距 | −0.300 → **+0.047** |

**統計邊界**:0/27 的 95% 上界仍有 **11%**。可寫「未觀察到碰撞」,不可寫「碰撞率為 0」。

## 下一步(依價值排序)

1. **100 趟驗證**(約 5.5 h)。0/100 → 上界 3%,那句話才能寫進論文。
   指令:`SHIELD=1 POSE_SOURCE=odom` + `bigarena_poses_big.csv`,對照用 `poseF`。
2. **補 shield 診斷欄位 `non_approach_ok`**。完整場景下 barrier 有 0.45% 的週期無解、
   fallback 啟動,但現有的 `unresolved` 量的是**完整 barrier** 的殘差,
   **沒有量到降級後的「不接近」條件是否守住**。程式上由結構保證,但今天已證明過
   數次「由結構保證」≠「實際發生」。
3. 第四次進度報告尚未寫入今天的內容(shield、速度閘門消融、質心速度偏差)。

## 今天新增/修改的東西

- `src/ammr_wholebody_mpc/.../scan_safety_shield.py` — 新節點,`SHIELD=1` 啟用
- `scan_obstacle_tracker.py` — 預測關聯 / Mahalanobis / 碎片合併 / coast,
  **全部預設關閉**(T1–T3 證實它們不是 seed27 的原因);新增 `/gmpc/tracks_debug`
  與 `_is_mover` 的拒絕原因碼
- `gmpc.py` / `gmpc_node.py` — 輪速多面體、`pose_source`、每週期 `/gmpc/diag`
- `generate_bigarena.py` / launch / `analyze.py` / `random_poses.py` — `SCENE` 參數化
  (預設 `bigarena`,種子 11 逐位元重現原場景)
- `evaluation/results/figs/bigarena.png` — 場景圖已更新為現行速度,標題「實驗場景」

## 相關結果檔

`SPEEDGATE_SUMMARY.md`、`SHIELD_SUMMARY.md`、`SHIELD20_SUMMARY.md`、
`OVERNIGHT_SUMMARY.md`,bag 存檔在 `evaluation/bags/archive_{S0,S1,U2,shield20,shield30,poseF,...}`
