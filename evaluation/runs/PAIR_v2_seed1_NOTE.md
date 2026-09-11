# 配對 v2（seed 1，bigarena，v3 排程，GUARD=1）

| | OFF | ON |
|---|---|---|
| RUN_ID | `v2_off_093455` | 待跑 |
| method | `gmpc_scan` | `gmpc_scan_heading` |
| HEADING | 0 | 1 |
| ROS_DOMAIN_ID | 81 | 83 |

## 兩趟相同（控制路徑、場景、相位）

| 檔案 | sha256 前 16 碼 |
|---|---|
| `gmpc.py` | `e2ded59dbe2a8b78` |
| `gmpc_node.py` | `b83d4d2072c7b240` |
| `wheel_limit_guard.py` | `e319aa88e7485642` |
| `dynamic_obstacle_driver.py` | `a3ad092d3152ce12` |
| `dynamic_trajectories_bigarena_traffic_v3.yaml` | `2e32069f3786b10d` |
| `omni_bot_dynamic.launch.py` | `d2f3a6439363cf08` |
| `isaac_bigarena_sim.py` | `364c3319c60557ee` |

`install/` 與 `src/` 逐檔相同，兩趟之間未重建。
設定亦相同：`AMMR_OBSTACLE_MODE=scheduled`、`PHASE_DELTA=30.0`、`GUARD=1`、
`CPU_LIMIT=92`、`CPU_THREADS=8`、`HEADLESS=true`、`CAMERA=false`、`RENDER_HZ=12`、
起點 (17.1, 14.8)、目標 (8.12, 0.48)、任務時限 180 s 模擬時間。

## 兩趟不同：**只有 bag 錄製清單**

ON 之前在 `evaluation/run_bigarena_isaac.sh` 加錄三個 topic：

* `/wheel_guard/status` —— guard 逐筆 `mode`／`action`／`lam`／`dt`／`faults`／`accel_guaranteed`
* `/cmd_vel_smoothed` —— smoother 輸出，即 guard 的輸入
* `/gmpc/diag_v2` —— 版本化診斷（`schema='gmpc_diag/2'`，帶 `cmd_id`）

同時把這三項加入「bag 已訂閱」就緒閘。這是**純錄製變更**：不經過控制路徑，
不改任何節點程式，不需要重建，不能影響機器人的行為。但它確實使兩趟的
錄製設定不同，因此在此明列。

## 比較規則（因上述差異而必須遵守）

1. 只比較**兩趟共同具備**的指標：到達與否、同定義的到達時間、真實路徑長度、
   車頭與行進方向夾角、前進／後退／側移比例、動態與靜態最近距離、轉動量、
   最終命令的輪速合規。
2. guard 的**修改次數、縮放量（`lam`）與模式切換只對 ON 報告**。
   OFF 沒有錄這些資料，**不可把 OFF 的缺失當成「零次」**。
3. OFF 的「最終命令值都能在控制器輸出中找到」是數值吻合，不是逐筆來源對應；
   ON 有 `/cmd_vel_smoothed` 與 `cmd_id` 才能做逐筆對應。兩者不可並列比較。
4. 輪加速度：OFF 只能報基於既有時間戳的命令序列估計；ON 才有 guard 的逐筆 `dt`。

詳見 `isaac_bigarena_gmpc_scan__seed1__v2_off_093455/DATA_SCOPE.md`。
