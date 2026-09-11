# 凍結設定：seed 1 配對之後的多 seed 批次（2026-09-11）

本批次**不調任何參數**。以下全部與 `v2_off_093455` / `v2_on_154856` 相同。

## 控制路徑（sha256 前 16 碼）

| 檔案 | sha |
|---|---|
| `src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py` | `e2ded59dbe2a8b78` |
| `src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_node.py` | `b83d4d2072c7b240` |
| `src/ammr_wholebody_mpc/ammr_wholebody_mpc/wheel_limit_guard.py` | `e319aa88e7485642` |
| `src/ammr_bringup/ammr_bringup/dynamic_obstacle_driver.py` | `a3ad092d3152ce12` |
| `src/ammr_bringup/config/dynamic_trajectories_bigarena_traffic_v3.yaml` | `2e32069f3786b10d` |
| `src/my_omnibot_description/launch/omni_bot_dynamic.launch.py` | `d2f3a6439363cf08` |
| `evaluation/isaac_bigarena_sim.py` | `364c3319c60557ee` |

`install/` 與 `src/` 逐檔相同。**批次期間不得重建。**

## 執行設定

```
AMMR_OBSTACLE_MODE=scheduled
AMMR_TRAJ_FILE=AMMR_PHASE_YAML=dynamic_trajectories_bigarena_traffic_v3.yaml
PHASE_DELTA=30.0   GUARD=1
CPU_LIMIT=92  CPU_THREADS=8  HEADLESS=true
CAMERA=false  RENDER_HZ=12  VIEW_WIDTH=1600  VIEW_HEIGHT=900
任務時限 180 s 模擬時間   arrive_tol 0.30 m
physics_dt = rendering_dt = 0.01
```

## 錄製（兩邊皆同，與 seed 1 的 OFF 不同之處）

seed 1 的 OFF（`v2_off_093455`）**沒有**錄 `/wheel_guard/status`、
`/cmd_vel_smoothed`、`/gmpc/diag_v2`。**本批次兩邊都錄**，因此：

* 本批次內部的 OFF／ON 可以在 guard 狀態與中間命令上直接比較。
* **與 seed 1 的 OFF 比較時，這三項仍不可比**（見
  `runs/isaac_bigarena_gmpc_scan__seed1__v2_off_093455/DATA_SCOPE.md`）。

## seed 選定（事前，非事後挑選）

判準：等待 `PHASE_DELTA = 30 s` 期間起點的動態淨距 ≥ 0.10 m，
起點與終點的靜態淨距 ≥ 0.10 m；淨距 = 中心距 − 障礙半徑 − 機器人半徑 0.30。
30 個 seed 全數通過（最小 0.643 m，seed 21），**v3 的相位沒有困住任何 seed**。

在通過者之中依「**跨起終點**」挑分散的 4 個，不就近取樣：

| seed | 起點 | 終點 | 行進方位 | 直線 m | 起點動態淨距 |
|---|---|---|---|---|---|
| 1（已完成） | (17.10, 14.80) NE | (8.12, 0.48) | −122.1° | 16.90 | 2.832 |
| 7 | (1.82, 17.15) NW | (9.64, −0.26) | −65.8° | 19.09 | 2.745 |
| 8 | (12.00, 0.45) S | (11.28, 15.38) | +92.8° | 14.95 | 3.064 |
| 13 | (0.54, 7.63) W | (8.88, 16.44) | +46.6° | 12.13 | 1.690 |
| 24 | (16.47, 1.57) SE | (0.47, 17.16) | +135.7° | **22.34** | 6.054 |

方位橫跨約 250°，起點分布四象限，長度 12.13–22.34 m。

## 執行順序（交替，避免順序效應與熱累積偏向同一邊）

| # | seed | 先跑 | domain | | # | seed | 後跑 | domain |
|---|---|---|---|---|---|---|---|---|
| 1 | 7 | OFF | 86 | | 2 | 7 | ON | 87 |
| 3 | 8 | **ON** | 88 | | 4 | 8 | OFF | 89 |
| 5 | 13 | OFF | 90 | | 6 | 13 | ON | 91 |
| 7 | 24 | **ON** | 92 | | 8 | 24 | OFF | 93 |

## 處置規則

* **正常逾時（`task_timeout`）或未到達，照實保留**，不重跑、不調參。
* 只有**啟動流程失敗或資料無效**才另行處理，並標記 `ABORTED.md`。
* 每組配對都要驗證**實際障礙軌跡**（驅動目標對獨立重算的排程、
  以及兩趟以各自相位零點對齊後的位置差）。

## 要回答的問題

**朝向改善是否能跨起終點維持？是否伴隨更小間距、更多轉動或較長任務時間？**

加上 seed 1 共 **5 組探索性配對**。n 仍小，重複試驗的波動尚未量過。
