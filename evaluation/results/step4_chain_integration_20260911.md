# 新版本整合驗收：命令鏈（2026-09-11）

**定位：命令鏈整合驗收。不接續舊 OFF／ON 配對，不是導航效能測試。**
無障礙開闊區域（`--empty-world true`）、相機關閉、獨立 `ROS_DOMAIN_ID=79`。
兩種故障分開跑，各自從靜止初始化。

## 鏈路與版本

```
step4_chain_driver → /cmd_vel_nav → velocity_smoother → /cmd_vel_smoothed
                   → wheel_limit_guard → /cmd_vel → Isaac
```

雜湊見 `versions/snapshot_guard_impl.json`（install 與 src 三個檔案皆一致）：
`wheel_limit_guard.py` `e319aa88e7485642e33cd6eb`、
`gmpc.py` `e2ded59dbe2a8b78cbb7b041`、`gmpc_node.py` `b83d4d2072c7b24066fb359f`。

設定：`input_timeout=0.25`（guard）、`--cmd-timeout 0.5`（Isaac 接收端）、
smoother 20 Hz／OPEN_LOOP／每軸 `max_accel [6.25,6.25,25.51]`。
輪級限制 `ω_max=5.55 rad/s`、`α_max=125 rad/s²`。

**停止判定（事前固定）**：真值平移速度 ≤ **0.01 m/s** 且持續 ≥ **1.0 s** 模擬時間；
真值取樣為模擬器 20 Hz 位姿發布。

## 故障 A：guard 的輸入停止更新，guard 保持運作

**注入方式**：終止 `velocity_smoother`——即 **guard 實際訂閱的輸入來源**。
只停 GMPC 並不足夠：本測試中驅動節點持續發布 `/cmd_vel_nav`（注入後仍有 305 則），
若只停上游，smoother 仍會持續發布而 guard 的逾時不會觸發。

| 項目 | 結果 |
|---|---|
| `/cmd_vel_smoothed` 注入後 | **0 則**（輸入確實中斷） |
| `/cmd_vel` 注入後 | **288 則**（guard 持續發布） |
| `input_timeout` 首次觸發 | 注入後 +0.494 s |
| **輪加速 max（輸出序列獨立驗算）** | **111.0000 / 上限 125.0，超出 0 / 593** |
| 輪速 max | 5.5500 / 上限 5.55，超出 0 |
| guard 模式 | 全程 `normal`（594 則） |
| Isaac 接收端逾時 | **未觸發**（正確：`/cmd_vel` 持續更新） |
| 真值首次低於門檻 | 注入後 +0.353 s |

**驗算方式**：以 `/wheel_guard/status` 的 `seq` 配對前後輸出，
用 guard 自己記錄的 `dt` 計算 `|Δω|/dt`，**不只看 `accel_guaranteed=true`**。
593 筆全部守限。

## 故障 B：guard 停止輸出

**注入方式**：終止 `wheel_limit_guard`。Isaac **持續步進**，
**未**以 cleanup、暫停物理或其他節點送零代替。

| 項目 | 結果 |
|---|---|
| `/cmd_vel` 注入後 | **1 則**（隨即停止更新） |
| 歸零前最後命令 | `[0.1721, 0.0, 0.4302]` |
| **接收端逾時觸發** | 模擬 37.270 s，注入後 **+0.541 s**，逾時長度 0.510 s |
| 真值首次低於門檻 | 注入後 +0.551 s |
| 輪速 / 輪加速（guard 輸出，注入前） | 5.5500 / 0.0000，皆未超出 |

**接收端緊急歸零是失效處置，不宣稱滿足正常加速度限制**——
Isaac 直接把 `node.cmd` 設為零，沒有經過任何限制。

## 一項量測限制（照實記錄）

注入時刻是**牆鐘**，其餘量在**模擬時間**。兩者的映射由 `/model/omni_bot/pose`
同時帶有的牆鐘與模擬時戳線性擬合而得，**殘差 118–139 ms**
（位姿取樣間隔本身就是 50 ms）。

因此：
- 故障 B 的「逾時觸發 +0.541 s → 真值首次低於門檻 +0.551 s」相差 10 ms，
- 故障 A 的「首次低於門檻 +0.353 s vs `input_timeout` +0.494 s」相差 141 ms，

**兩者都落在映射殘差之內，不足以判定先後順序**。各自的絕對時刻與相對注入的
時間差仍有效；**只有「誰先誰後」這個判斷無法由本測試支持**。
先前一版分析把「停止時刻」算成 `sim[i] - hold`（持續判定的起點往前減去 hold），
那是錯的，已改為分別回報「首次低於門檻」與「通過持續判定的該段起點」。

## 結論與保證範圍

> **故障 A**：guard 的輸入中斷後，guard 持續發布並把命令帶到零；
> 輸出序列以 `seq` 配對獨立驗算，輪速與輪加速度全程守限（0/593 超出）。
>
> **故障 B**：guard 停止輸出後，Isaac 接收端的獨立命令逾時於 0.510 s 觸發並歸零，
> 機器人停止；該次歸零是**失效處置**，不宣稱滿足正常加速度限制。

**不保證**：
- 實際輪子的物理加速度（本鏈只界定命令）；
- 任意排程條件下都不超限——本測試是固定 20 Hz 節奏；
- `--cmd-timeout 0.5` **是本測試的設定值，不是經過論證的安全停止時限**；
- smoother + guard **最終命令**的 CBF 配對診斷**尚未完成**
  （`cbf_resid_evaluated_on='returned'` 只涵蓋 solve 的回傳命令）。

## 資料

`evaluation/runs/step4_smoother_091812/`、`evaluation/runs/step4_guard_092022/`
（各含 `chain.json`、`inject.json`、`isaac_run.json`、各節點 log）。
分析腳本 `evaluation/analyze_step4.py`。
