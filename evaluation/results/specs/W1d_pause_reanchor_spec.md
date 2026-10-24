# W1d：主動保持段伺服錨點一次性重設（單一變因候選）——差異規格＋實作＋測試＋凍結 w1d r1（**未開跑**）

依 Codex 20261025_010414：
- 同意將「保持段伺服錨點一次性重設」做成版本化規格＋實作＋測試；
- 沿用 W1c r2 其餘全部設定（含偏移估計保持）；
- 本輪尚不核可實跑；一次送差異規格、實作、測試與新凍結，再申請單趟 30°；
- 保留全部停止判準與 5 mm 幾何門檻；不新增補償、不推進 W2。

本輪要檢驗的問題：**重設伺服參考後，命令、相對位移與淨空下降是否改變。**
這是單一變因候選，不是已確認的解法。

## 0. 依據（W1c_30_1_pause_geom_analysis.md r2，只描述）
- door_wide_w1c_30_1：門開到約 30°（SETTLE 29.87°），命令停止已確認（STOPPED 56.67 s），62.00 s 被幾何監看中止。
- 暫停期間殼—桿淨空下降 1.61 mm，伴隨門續轉約 0.0005 rad/s（只固定門角時 −0.01 mm，只固定機器人時 −1.60 mm）。
- TCP 相對把手沿 +y 位移 1.64 mm（這不等同於「橫桿相對夾爪」）；finger2 切向承載接近摩擦上限（μd·N 比值只作幅值參考）。
- 減速段末尾設定的保持錨點，與底盤靜止位置差約 1 mm。
  - 本趟實測：首次 ZERO_PUB 時位姿減錨點為 (−0.90, −0.87) mm，偏航 −0.59 mrad；到 61.9 s 仍為 (−0.95, −0.84) mm、−0.51 mrad。
- 因此伺服恆定推送約 2.4 mm/s（轉到把手座標為 +y，與相對位移同向），而底盤受夾持約束不動。
- 「同向」不等於因果。

## 1. 唯一差異
| 項目 | W1c r2 | W1d |
|---|---|---|
| 求解節點 | ADM9 | **ADM10**（ADM9 只插入，原行改動 0；`mk_adm10` 核對 2454 行依序全在 2496 行中）＋`--pause-hold-reanchor --pause-reanchor-max-age-s 0.2` |
| PAUSE 內部底盤保持伺服錨點 | 減速段末尾（decel→hold）設定一次，之後不變 | 同左，另在**首次 ZERO_PUB**那一輪以實測底盤位姿重設**一次** |

其餘完全相同：
- 任務：θ 30°、OPEN 逾時 52 s。
- 保護範圍：延伸走廊表與內帶、包絡 r2、觀察器 34°、5 mm 幾何門檻、全部停止判準。
- 控制：引導 w_vref 1、p4b 暫停（手臂 0.5 s 窗口）、偏移估計保持、權重、伺服增益與限幅（k_p 2.0、v_max 5 mm/s、w_max 0.02）、夾持力、摩擦。

任務節點、執行端、觀察器、走廊與包絡基準都不改（同一檔案、同一雜湊）。

## 2. ADM10 重設語意（純邏輯在 `door_k3m_pause_reanchor.HoldReanchor`）
**範圍**
- 只改 PAUSE 內部伺服錨點 `self._pause['anchor']`。
- **不動**：停車閘門錨點（`/park/gate` anchor）、走廊／包絡基準、凍結 TCP 目標、d̂。

**時機**
- 監督器首次回報 ZERO_PUB 的那一輪（`_act4['zero']` 且之前 `zero_mode` 為假）只呼叫一次。
- ZERO_PUB 屬於主動保持段，底盤仍是錨點伺服，**不是零命令**。

**來源**
- 同輪快照的實測底盤位姿 `snap.q[:3]`，即求解所用的同一快照（/odom 同時刻配對）。
- 新鮮度：`sim_now − snap.t_base ∈ [0, 0.2] s`。

**優先順序**
- 已收到停止請求，或監督器已 FAIL ⇒ 不重設（記為 `skipped:aborted`）。
- 量測非有限、長度錯誤或缺失 ⇒ 拒收（`rejected:pose_invalid`）。
- 量測過期或時間在未來 ⇒ 拒收（`pose_stale`）。
- 舊錨點缺 ⇒ 拒收（`old_anchor_missing`）。
- 拒收時保留舊錨點，之後**不再嘗試**。
- 重送或重入都不會再重設（只取第一次）。

**生效時點（照實登錄）**
- 邊緣那一輪的保持命令已用舊錨點算好並發布。
- 新錨點自**下一次求解**的保持目標起生效：
  - `effective` 記錄第一個用新錨點算保持目標的 slot、sim_t、目標值；
  - `first_published` 記錄生效後第一筆已發布解（slot、source_seq、publish_sim_t_exact）。

**承接與限制**
- 保持命令仍是 `decel_step(u_prev, goal)`，承接 applied u_prev。
- 既有加速度框、輪級與發布可行性核對（`publish_feasible`）照舊；不另設平滑，也不放寬。
- 既有加速度框為 a_base_lin·dt＝0.50×0.05＝0.025 m/s／週期，大於保持界 5 mm/s。因此重設後一週期內命令即可回到約 0，這仍在既有框內。

**紀錄**
- `stats.pause_reanchor`：record（old／new／anchor_shift、edge_sim_t、source_step、source_sim_t、age_s、status、why、effective、first_published）、n_calls、max_age_s。
- `k3m_pause.events` 增加一筆 `stage: reanchor`。
- 啟用時，每列另記 `pause_anchor_used`（本輪保持目標實際用的錨點）與 `park_anchor`；邊緣列記 `pause_reanchor_edge`。

## 3. 判定器 w1d（w1c r2 全部照用＋重設證據；缺證據不通過）
- **A1**：status＝applied、n_calls＝1、old／new 有限。
- **A2**：old＝唯一一筆 hold 事件的錨點；恰一筆 reanchor 事件，且與紀錄一致。
- **A3**：new＝room_run 來源步（sim_t 差 ≤ 1e-6 s）的實測 base_xyth（≤ 1e-5）；age 在 [0, max_age]。
- **A4**：恰一列邊緣紀錄，且即第一列 ZERO_PUB；該列仍用舊錨點。
- **A5**：effective.slot＞邊緣 slot；自該 slot 起用新錨點，之前沒有任何列用新錨點。
- **A6**：first_published 精確對上真實已發布列（slot、序號、發布時間）。
- **A7**：監控參考未移動——每列 park_anchor 恆定且＝park_gate.json gate.anchor；凍結 TCP 目標存在。d̂ 由 w1c H1–H6 核對。
- **A8**：暫停期每列發布可行。
- **A9**：求解端 args 的 pause_hold_reanchor 與 max_age 等於登錄值。

## 4. 測試（系統 Python，全過）
| 測試 | 項目 |
|---|---|
| `test_door_k3m_pause_reanchor.py` 22 項 | 未啟用；非零錨點誤差 applied、shift＝new−old（含 yaw 包角）；重送不再重設；effective／first_published 只記第一次且有先後；中止優先、中止後不再嘗試；缺量測／NaN／inf／長度錯誤／非數字拒收；過期、未來時間、NaN 時間拒收；邊界 0.2 s 接受；舊錨點缺拒收；拒收後不登錄生效 |
| `test_dl3_wgmpc_wg2_adm10_reanchor_ros.py` R0–R3 | 假世界（free 3.0 s、pause 5.0 s，W1c 偏移保持開啟）。替身 `wgmpc_sp_stub_endpoint_w1d.py`（原替身只插入 `--odom-dx-at/--odom-dx`）在 5.6 s 起把 /odom x 改報 3 mm，製造**非零錨點誤差**。**R1**：恰一次 applied；old＝hold 錨點 (0,0,0)、new＝(0.003,0,0)；邊緣 slot 111 仍用舊錨點，effective slot 112（下一次求解）；之後 202 列都用新錨點、之前 20 列沒有；first_published 對上真實已發布列；重設前命令 −5 mm/s（伺服限幅）、重設後目標 0；已發布底盤命令逐週期變化 ≤ 0.025 m/s；每列發布可行；park_anchor 不變；d̂ 保持；暫停照常 DONE。**R0** 未啟用：無紀錄、無新欄位（ADM9 行為），DONE。**R2** ZERO_PUB 前停止：stop_topic、無 applied。**R3** 缺 `--pause-active-hold` ⇒ 參數錯誤 |
| `test_door_k3m_check_w1d.py` 15 項 | 夾具＝R1／R0 實際求解紀錄（`results/door/fixtures/w1d_adm10_fakeworld_r{1,0}.json`）：一致 ⇒ 通過；以下各自不通過——R0 或缺紀錄、rejected 或 n_calls 2、old≠hold 錨點、兩筆事件、來源位姿不符或缺來源步、age 超限、邊緣列假裝已生效、effective＝邊緣、生效後某列用舊錨點、序號被改、park_anchor 移動、發布不可行、凍結目標缺、登錄未含 pause_hold_reanchor、args 與登錄不符 |

替身沒有物理，只核接線、語意與紀錄；夾持中的底盤反應只能在 Isaac 實跑觀察。

## 5. 風險（揭露）
- 錨點重設後，若底盤有漂移傾向，伺服須重新累積誤差才回推，期間可能出現短暫的底盤速度。
  - 監督器的 moved 判準與 5 mm 幾何門檻照常會中止。
  - p4 時我曾因這個顧慮沒有採用錨點重設。
- 本輪不能證明以下任何一項：
  - 錨點偏移推送造成滑動（同向不等於因果）；
  - 錨點重設能讓實際停止成立；
  - 淨空下降只由底盤命令造成（門續轉同樣伴隨）。

## 6. 申請單趟 `door_wide_w1d_30_1`（凍結 w1d r1）
- 乾淨環境、92 °C；失敗保留不補跑。
- **分報**：
  - 開門（θ）、SETTLE、命令停止、實際停止、保護、偏移估計保持證據；
  - 重設證據（old／new／shift、生效 slot、首筆發布）；
  - 重設前後的底盤保持命令、底盤實測速度、TCP 相對把手位移、殼—桿淨空、門續轉速率。
- 對照 w1c_30_1 暫停期，只作單趟描述，不作統計結論。
- 附 side／close 影片。
