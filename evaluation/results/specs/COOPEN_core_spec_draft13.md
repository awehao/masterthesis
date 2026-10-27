# 連續協同通行：核心段 — 規格 draft-13（台架初態啟動，bench5 版）

draft-12（bench4）保留。本版只修正 `door_coopen_core_bench_2` 暴露的第二個啟動順序缺口（Codex r24）；**其餘全同 draft-12**。

## 1. bench_2 封存
- 封存為「首解未送達執行端，物理未釋放，核心動作未受測」。
- 錨點等待生效（0.0 s）；預熱 71 ms；首解宣告 seq 2。
- 執行端凍結期收到的轉接器段封裝為 0 筆，120 s 逾時、不釋放。
- 主要解釋：轉接器先做順序核對，10:31:26.45 才建立訂閱，首解在訂閱建立前發出而遺失。
  - 尚未排除 yaw 新鮮度拒收的可能。

## 2. 修正（只改啟動順序；不加固定 sleep、不放寬 yaw 年齡、不提前發布）

### 轉接器 `arm_vel_adapter_k3m2.py`（k3m 保留）
- 帶 `--bench-ready-run-id` 時，每 0.05 s 發 `/bench/adapter_ready`，內容：run_id、遞增 seq、order_ok、yaw 有限與年齡、n_pub_in、n_sub_out、ready。
- **ready 須同時成立**：
  - 順序核對成功；
  - yaw 有限且年齡 ≤ 0.3 s（與轉送判準同一條）；
  - 求解端→轉接器、轉接器→執行端兩段訂閱都已匹配。
- 轉送行為不變；旗標未給時不建立回報。

### 求解端 ADM17r3（r2 保留）
- 預熱之後、正式首解之前，有界等待轉接器就緒：
  - 本趟 run_id、ready 為真、seq 遞增、本地收到 ≤ 0.2 s；
  - 本節點封裝話題的訂閱者 ≥ 1。
- 上限 30 s，逾時拒跑。結果記入 bench.adapter_ready。

### 跑法 bench5
- ＝ bench4，只換轉接器、求解端與凍結。

## 3. 測試（test_door_bench_start_ros3：**真 ADM17r3 ＋ 真轉接器 k3m2 ＋ 替身執行端 r3**；替身只接轉接器段封裝，釋放與首步核對用真 EnvIntake／E2／DualSourceExecutor）

| 情境 | 設定 | 結果 |
|---|---|---|
| C1 延後訂閱 | 轉接器晚 3 s 啟動 | 等待 3.35 s 後首解；同 seq 釋放；first_intake 與 first_step 皆 ok |
| C2 延後 odom | 前 5 s 不發 odom | 5.11 s 釋放；同 seq；首步 ok |
| C3 yaw 過期後恢復 | odom 1–7 s 停 | 7.15 s 釋放；同 seq；首步 ok |
| C4 缺就緒逾時 | 不啟動轉接器、等待 4 s | 求解端拒跑；無宣告；未釋放 |
| C5 旗標關閉（靜態） | — | k3m2 對 k3m 只有新增；就緒計時器只在旗標時建立 |

- **揭露**：C2、C3 中求解端自身的狀態握手也需要 odom，所以就緒等待時 odom 已恢復（等待 0.008／0.04 s）。這兩例主要驗證「odom 不在時不發出首解」。

## 4. 實跑（Codex r24 條件式核可：上述全過、新凍結後一次）
- 單趟 `door_coopen_core_bench_3`，跑法 bench5，凍結 `freeze_door_coopen_core_bench_dev_r5`。
- 範圍與條件同 draft-11。
