# K3-M 受控暫停接入規格 draft-1（Codex 20261024_014938；版本化實作＋離線／ROS 假世界測試；**不核可實跑**）

**範圍**：
- K3-M（底盤可動小角度協同開門）的**健康狀態正常結束**，改用受控暫停取代「END_HOLD 3 s → DONE 停求解器 → 命令逾時」。
- 異常一律走既有 K2 閂鎖，不改。
- 正常暫停通過**不等於**異常停止風險解除。

## 1. 單一控制權與狀態轉換
```
OPEN ──(SETTLE 完整健康條件連續 1 s)──▶ PAUSE_REQ ──(三方握手 seq=2, mode=pause 確認)──▶
  pause-decel ──(實測底盤 ≤ 伺服界)──▶ pause-hold ──(實際低速連續 1 s)──▶ ZERO_PUB ──(發布底盤命令＝0)──▶
  CONFIRM ──(套用回報：5 個連續不同實際物理步、來源序號 ≥ 零速第一筆、套用值 ≤ 1e-6)──▶ STOPPED（停止發布）──(觀察 10 s 全程實際低速)──▶ DONE
任一時刻：K2 中止（任務／執行端／觀察器）優先於所有待處理切換 ⇒ 既有閂鎖；暫停失敗（moved_*／timeout／solve_failed／measurement_invalid）⇒ 記錄後走 K2 閂鎖
```
- **單一控制源**：
  - pause-decel 與 pause-hold 的底盤命令都由 **W-GMPC 求解端**的 `cfg.base_hold` 等式產生（同一求解、同一發布）。
  - decel 以 `decel_step(applied u_prev, 0)`；hold 以 `decel_step(applied u_prev, park_hold_cmd(anchor))`。
  - 不另起停車伺服節點；執行端不產生底盤命令。
- **手臂**：W-GMPC 追凍結目標（`PauseReference`：最後一筆已發布目標，含 seq／t），近目標整形維持停用。
- **ZERO_PUB 起**：發布值＝底盤 decel 朝 0、手臂 0。STOPPED 起停止入列；在途命令照常套用，年齡 > 0.2 s 由 E2 既有逾時處置。
- **握手**（新模組 `door_k3m_pause_mode.py`，沿用 door_k3m_mode 的規則）：
  - 任務端提出 {run_id, seq=2, mode='pause', req_step, req_t, frozen_target_seq}；
  - 執行端為權威：須已在 free（seq=1 已切換）、未中止 ⇒ ack {switch_step, switch_t}，否則 nack；
  - 求解端只認執行端 ack，第一筆**已發布**的 pause 解才回 ack；
  - 任務端核對兩 ack 一致，且在雙期限內（模擬＋牆鐘）才確認；
  - 中止優先於冪等重送。

## 2. 完整健康條件
- **P1 觸發**：沿用 K3-A 正式 SETTLE（`door_k3a_policy`）全部條件連續 1 s：
  - |q − 8°| ≤ 0.5°、|q̇| ≤ 0.002；
  - 抓取漂移 ≤ drift_abort_m；
  - 兩指橫桿接觸 ≥ contact_min_n；
  - 以及策略每週期都跑的 ACTIVE_MONITOR：狀態／接觸／求解新鮮度、非設計接觸、牆接觸、腕力、門角與門速界、K3-M 幾何監看、走廊與包絡（執行端）。
- **暫停全程**（PAUSE_REQ 到 DONE）：上述監看持續；`PAUSE` 加入 ACTIVE_MONITOR 與 FINGER_BAR_ALLOWED。任何違規 ⇒ K2 閂鎖。

## 3. 命令停止與實際停止分報（`door_k3m_pause_sup.PauseSupervisor`，已實作，反例 7）
- **命令停止確認**：到 STOPPED（零速命令經套用回報確認、停止發布）。
- **實際停止確認**：
  - 量測：同步狀態（50 Hz）的底盤位姿差分、門角差分、實測關節差分，以 0.1 s 窗口估速度；缺樣本或間隔 > 35 ms ⇒ 無值。
  - 門檻：與長窗判準相同（1 mm/s、0.005 rad/s、0.002 rad/s、1e-3 rad/s）。
  - **自 ZERO_PUB 起**，CONFIRM、停止發布後的 **10 s 觀察**全程都要低速，否則 `moved_*`。
- **判定用途**：只作完成與評估判定，**不替換原門速安全保護**。
- **整體暫停成功**＝實際停止確認；觀察未滿 10 s、再動、保護提前終止 ⇒ 不得報成功。
- **離線判定器另以 room 逐步紀錄獨立重算**，不採信線上結果。
- **兩個有限情境**（取自 lw_1 實錄，不是門動力學模型）：門獨立續轉、零命令底盤漂移，都不會誤報停止（已測）。

## 4. 來源序號穿透管線（`--cmd-env`；三端擇一入口，既有設計）
- **求解端**：ADM4 發布封裝 `/wgmpc/cmd_env`（source_seq 逐筆遞增）。
- **adapter**：`arm_vel_adapter_k3m.py`，arm_vel_adapter 的版本化複本，只新增 `--env-in-topic`，預設＝原 ST_SAFETY；K3-M 無安全層，指向 `/wgmpc/cmd_env`。轉換邏輯不變。
- **執行端**：`isaac_door_k2_sim_ev4.py`，ev3 的只插入複本。
  - `--cmd-env` 時不建立 `/wb_vel_cmd` 訂閱，只收 `/wb_vel_cmd_env`；以 E2 鏈 `recv_seq` 對照 source_seq。
  - 每個實際物理步發布 `/door/applied_src`：[step, sim_t, source_seq（逾時或停止為 −1）, 鏈模式（0 正常、1 逾時、2 停止、3 失效）, 底盤 3, 手臂 6]，並寫入 room steps 新欄 `applied_src`。
- **求解端確認**：只用 `/door/applied_src` 的 source_seq。舊命令即使數值為零，也因序號 < 零速第一筆而不計；停止發布後在途舊命令若重新生效（序號倒退）⇒ 失敗。
- **測試**：延遲、重送、停更、中止優先、舊命令不得在零速確認後重新生效（ROS 假世界，使用真 CmdChainE2 的鏈替身）。

## 5. 差異表（相對 K3-M r3 凍結的跑法）
| 檔 | 變更 |
|---|---|
| `door_k3m_pause_sup.py` | 新（命令／實際停止分報）＋測試 |
| `door_k3m_pause_mode.py` | 新（seq=2 pause 握手）＋測試 |
| `arm_vel_adapter_k3m.py` | adapter 複本＋`--env-in-topic` |
| `isaac_door_k2_sim_ev4.py` | ev3 只插入：`--cmd-env` 封裝入口、`/door/applied_src`、room 欄 `applied_src`、pause 握手執行端權威 |
| `dl3_wgmpc_wg2_adm4_node.py` | ADM3 只插入：pause 模式（凍結目標、base_hold decel／hold、ZERO_PUB／停止入列、PauseSupervisor、`/wgmpc/pause_status`） |
| `door_k3m_policy_p.py`、`door_k3m_task_node_p.py` | K3-M 策略與任務節點複本：SETTLE 完整條件 → PAUSE_REQ／PAUSE 相位、全程監看、暫停結果收尾 |
| `door_k3m_check_p.py` | 判定器複本：加暫停分報（獨立重算命令與實際停止、觀察覆蓋、re-motion） |
| `run_door_k3m_pause.sh` | 跑法：`--cmd-env` 三端、獨立觀察器（門角範圍上限改為 10°）、92 °C、凍結 |

## 6. 首趟（日後申請，本輪不跑）
- **範圍**：單趟「小角度（8°）協同開門＋受控暫停結束」。
- **保護**：既有閂鎖＋觀察器（非設計接觸或牆 > 0.1 N、腕 ≥ 30 N、門角 ≥ 10°、門速 > 0.05、停更 0.2 s），以及 92 °C；觀察上限＝暫停 STOPPED 後 10 s。
- **結束模擬不算實際停止**。
- 結果分報：任務、暫停命令停止、暫停實際停止、保護完整性；附影片（該趟實錄重播）。
