# coman_drawer20_main4：**九筆命令發出、執行端一筆未收**（QoS 不相容）

**未取得操作與同動結果；不算單步 QP 任務失敗。**
這一趟第一次真的解出並發出九維命令，但命令在 `/wb_vel_cmd` 上因 QoS
不相容全部落地，執行端 `received = 0`、`coman_applied_cmd_log` 長度 0
——**沒有任何一筆命令被套用到模擬**。因此三題（完整操作、拉動期間同動、
滿載時效）都**沒有證據**，不能拿本趟填任何一題。

| 項目 | 值 | 來源 |
|---|---|---|
| 起動前檢查 | `{"failed": []}` | `preflight.json` |
| 底盤模式 | `free_base`，`root_fixed_joints = []` | `sim/drawer_run.json` |
| 目標行程 | `target_opening_used_m = 0.020`（案例值 0.200 未被沿用） | 同上 |
| 求解週期 | **9 筆**，相位**全部 APPROACH** | `solver_out.json` |
| 命令送達 | **`chain9.received = 0`、`rejected = 0`** | `sim/drawer_run.json` |
| 套用命令 | `coman_applied_cmd_log` **長度 0** | 同上 |
| 連接 | `attached_ever = false` | `coman_couple_link` |
| 最終開度 | **0.00 mm（誤差 −20.00 mm）** | `final_opening_m` |
| 停止原因 | `sim_limit` @ sim 120.010 | `stop_reason` |
| RTF（實測） | `sim_time_s 120.61 / wall_s 120.12` = **1.004** | 同上 |
| 溫度 | 起 56.6 → 峰值 **79.625 °C** @ sim 11.82 → 收 77.25（限 **92**） | 同上 |
| 監看 | `monitor_failure = None`、`coman_abort = None` | 同上 |
| 腕力峰值 | 6.016 N @ 相位 `idle`（中止線 30 N 連續 0.05 s） | `f_norm_peak` |

## 主因（F14）：`/wb_vel_cmd` 的 QoS 不相容

**兩端都各自記錄了**，訊息在 run.log 裡是明文：

```
行432  arm_vel_adapter
  New subscription discovered on topic '/wb_vel_cmd', requesting incompatible
  QoS. No messages will be sent to it. Last incompatible policy: DURABILITY
行438  isaac_drawer_sim
  New publisher discovered on topic '/wb_vel_cmd', offering incompatible
  QoS. No messages will be received from it. Last incompatible policy: DURABILITY
```

鏈的方向是 `arm_vel_adapter`（發布，VOLATILE）→ `/wb_vel_cmd` →
`isaac_drawer_sim`（訂閱，TRANSIENT_LOCAL）⇒ **DURABILITY 不相容，
整條完全落地，不是偶發遺失**。這與 F10（diag／cmd_meta）是**同一類錯誤
在另一個主題上重演**：F10 修了兩個診斷主題，卻沒有一併掃過命令主題。

九筆命令的內容本身是正常的（底盤三維與手臂六維同時非零，底盤橫向分量
達 −0.2775 = `base_lin` 飽和值）：

```
seq=1  [ 0.0000, -0.2495,  0.2532,  0.0060,  0.9992,  0.9992, -0.0099, -0.9878, -0.0097]
seq=5  [-0.0000, -0.2775,  0.0185,  0.0396,  1.1376,  0.9558, -0.0703, -1.0051, -0.0701]
```

**但這不是同動證據。** 這些是 APPROACH 相位的**命令值**，既未送達也未套用，
而且同動要求的是**拉動期間**的實測連續同動。命令裡底盤與手臂同時非零，
只說明求解器輸出了九維解。

## 兩個獨立的執行期時效問題

### 距離節點：p50 97.96 ms，週期 **33.3 ms** ⇒ 實際約 10 Hz

> **更正（2026-10-01，main5 之後）**：此處原寫「週期 50 ms」有誤。
> 距離節點的 `publish_rate` 預設為 **30 Hz ⇒ 33.3 ms**；50 ms 是控制端
> （求解／安全，20 Hz）的週期。因此 main4 的 p50 是目標的 **2.94 倍**，
> 比原文所述更差。下方求解器與安全層的 50 ms 引用是正確的。

依 `~/diag_fields` 名單**按名稱取欄**（不依位置）：

| 欄 | n | p50 | p95 | max |
|---|---|---|---|---|
| `dist.node_cycle_ms` | 1078 | **97.96** | 110.56 | 170.74 |
| `dist.n_rows` | 1078 | 1350 | 1350 | 1350 |
| `dist.dup_dropped` | 1078 | 86 | 86 | 86 |
| `dist.worst_age` | 1078 | 0.00 | 0.09 | 0.12 |

1078 筆橫跨 sim 11.87–120.01（約 108 s）⇒ **約 10 Hz，不是 20 Hz**。
`n_stale` 全程 0、`worst_age` 最大 0.12 s，資料新鮮度本身沒有失效。
`min_d` 全程恆為 0.38 m 不變 —— 與「機器人從未移動」一致。

此項已在本輪離線處理：見 S1 的 `F14_followup_speedups`（52.52 → 26.64 ms，
同一批輸入逐位元等價）。**但那是 998 列的假世界量測，不能當作本趟
1350 列滿載的改善值**；線上數字必須下一趟重新量。

### 求解器：9 筆中 6 筆超過 50 ms

`solve_ms` = 133.17、2.60、2.93、2.43、115.20、128.37、133.58、106.16、100.15
⇒ p50 **106.16**、max **133.58**、**6/9 > 50 ms**。
9 筆**不足以刻畫分布**，但這 9 筆本身已明顯超出週期，且**距離節點的加速
完全不會改善它**——這是另一個獨立問題。九筆的模擬時間只從 11.91 走到 11.98。

### 安全層的 0.12 ms 是空轉成本，不是正常成本

`SafetyLike` 是「該週期沒有計算」時的零值替身。全部 2368 個週期中
**只有 21 個 `n_rows > 0`**：

| | 滿載（21 筆） | 空轉（2347 筆） |
|---|---|---|
| `node_ms` | p50 **26.30**，max **47.64** | p50 0.12，max 1.37 |
| `filter_ms` | p50 **14.67**，max **21.99** | 0.00 |
| `n_active` | max 212 | 0 |
| `iters` | max 75 | 0 |

`fallback` 與 `unresolved` 在這 21 筆全為 0。
滿載時 `node_ms` 已達 47.64 ms（週期 50 ms）。**21 筆不足以刻畫分布。**
安全層的 `n_rows`（2736／2754）是濾波器的**約束列數**，與距離節點的
雲列數 1350 **不是同一個量**，不可相減或相除比較。

## 求解端的守門中止：JointState 586 ms 未更新

```
中止（guard）：JointState 已 586 ms 未更新
9 週期寫入 solver_out.json
```

求解端在 sim ≈ 11.98 被自己的新鮮度守門擋下並收束；執行端則繼續跑到
sim_limit 120.010。**原因尚未確立**，列為 O7。已知的相關事實：

- `/joint_states` 是在**物理迴圈內同步發布**的（`isaac_coman_drawer_sim.py`
  約 2002 行），不是獨立的 spin 執行緒 ⇒ 不能用「發布執行緒被餓死」解釋。
- 全趟 RTF 1.004。**但這是 120 s 的平均值，無法排除一次 586 ms 的停頓**，
  不得拿來為物理迴圈開脫。
- 停頓時點（sim 11.9–12.0）正好是距離節點首次 `_tick`（sim 11.87，
  立即 98 ms/週期）與求解器首次求解（133 ms）同時上線的時刻；
  溫度峰值 79.625 °C 也發生在 sim 11.82。**時間吻合不等於因果**，
  尚未量測該時段的 CPU 佔用或求解端執行器的阻塞時間。

## 已生效的既有修正

F1（模式相依守衛，`free_base` 放行）、F2（兩端目標 0.020 m）、F3、F4、F5、
F6、F7、F9（求解端等到 OK 列才解，未立刻中止）、F10、F11、F12（距離節點
不再在第一個 `_tick` 死亡，連續跑滿 108 s）、G8a／G8b。R1.1 十五組間距
在兩端載入一致（run.log 行 **433**（`wholebody_safety`）與 **440**
（`wholebody_pregrasp`）的兩份 `g_by_pair` 各 15 組，**逐項相同**）。

F7 的生效證據：`safety.src_paired = 1` 出現 **5 個週期**，
`safety_meta.src_seq` 非零值為 {2, 5, 6, 7, 8} ⇒ `_on_src_meta` 確實掛上並觸發，
求解端→安全層的 `cmd_meta` 路徑通了 9 筆中的 5 筆。
**`cmd_meta` 與 `/wb_vel_cmd` 是不同主題** —— 前者通了不代表後者通了，
本趟正是前者部分通、後者全斷。

## 本趟新發現

| 編號 | 內容 |
|---|---|
| **F14** | `/wb_vel_cmd` QoS 不相容 ⇒ 命令全部落地（已修，離線測試見 S1） |
| **F15** | 封存紀錄的 `coman_stage_note` 與 `base_fixation.caveat` 是寫死的固定底座說明，與本趟 `mode = free_base` 相矛盾 ⇒ 已改為模式相依 |
| **O5** | 距離節點線上 p50 97.96 ms ＞ **33.3 ms** 週期（見上方更正）（等價加速已做，滿載值待量 —— main5 量到 p50 31.44 ms） |
| **O6** | 求解器 `solve_ms` p50 106 ms（9 筆），與 O5 獨立，**尚未處理** |
| **O7** | 求解端 JointState 586 ms 未更新，原因未確立 |
