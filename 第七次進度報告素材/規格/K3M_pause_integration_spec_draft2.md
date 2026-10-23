# K3-M 受控暫停接入 draft-2（已實作＋離線／ROS 假世界測試；凍結 k3m_pause r1；**未開跑**）

依 Codex 20261024_014938。draft-1（`K3M_pause_integration_spec_draft1.md`）的狀態轉換、單一控制源、完整健康條件、命令／實際停止分報、來源序號穿透都已實作。本文件列出實作中確定或修正的事項、差異表與測試。

## 1. 實作中確定的事項
1. **展開路徑與封裝入口並存**（ev4）：
   - 展開節點（UNFOLD）原本就經 `/wb_vel_cmd` 驅動手臂；求解端在 `--cmd-env` 下不走這條（adapter 封裝模式只發封裝）。
   - 因此 ev4 在 `--cmd-env` 時仍保留 `/wb_vel_cmd` 給展開節點，求解端命令只經 `/wb_vel_cmd_env`。
   - 「收到第一筆封裝後，舊路徑仍有命令」記為 `n_plain_after_env`，判定器要求 0。
   - 不在執行端丟棄這些命令，因為那會改變控制行為。展開節點的交棒用 `--solver-handshake`，不看發布者數量，不受影響。
2. **PAUSE 的監看**：
   - `door_k3a_policy` 的相位常數不能改，所以 PAUSE 期間監看與夾持檢查**以 END_HOLD 身分**執行，監看集合相同：門角界、雙指接觸、新鮮度、非設計／牆接觸、腕力、門速、K3-M 走廊與幾何。
   - 中止紀錄標實際相位 PAUSE（另記 monitored_as）。
   - 判定器把 PAUSE 對應為 END_HOLD 套用 K3-M 判準，所以 PAUSE 期間同樣要求完整幾何證據。
3. **凍結目標**：
   - 求解端凍結「話題收到的最新目標」；沒有話題目標時，凍結上一輪實際使用的目標。
   - ack 帶回凍結目標位置，任務端與判定器都核對它等於最後一筆已發布目標（請求的 frozen_target_seq 對到 trace）。
4. **停止發布後只觀察**：求解端在 STOPPED／DONE／FAIL 後不求解、不發布。這段放在時間上限、halt、鏈失效檢查**之後**、gate 之前（第一版放在迴圈最頂端，會跳過時間上限與 SIGTERM，由假世界測試抓出）。
5. **舊命令重新生效**：STOPPED 後若出現來源序號 ∈ [0, 零速第一筆) 或任何非零套用 ⇒ FAIL `old_command_after_stop`。來源序號 −1（逾時減速）只要值為零就接受。
6. **實驗保護**（同長窗版）：
   - 獨立觀察器 `door_k3m_observer.py`：長窗觀察器的複本，加上門角範圍上限參數，預設 12°。中止後生效：非設計／牆 > 0.1 N、腕 ≥ 30 N、門速 > 0.05、停更 0.2 s、無效資料。
   - 任務節點監看觀察器心跳：就緒前不推進，就緒後 > 1.0 s ⇒ 有界結束。
   - 跑法在觀察器起動 5 s 後未存活 ⇒ 拒跑；輸出機讀 `LW_HEALTH` 列。

## 2. 差異表（相對 K3-M r4 凍結的 run_door_k3m_ev.sh）
| 檔 | 性質 | 內容 |
|---|---|---|
| `isaac_door_k2_sim_ev4.py` | ev3 複本（原行改動 3 處：訂閱行加註解、room 欄與列的延續） | `--cmd-env`、封裝進鏈與 recv_seq→source_seq、`/door/applied_src`、room 欄 `applied_src`、暫停握手權威、`n_plain_after_env` |
| `arm_vel_adapter_k3m.py` | adapter 複本（3 行） | `--env-in-topic` |
| `dl3_wgmpc_wg2_adm4_node.py` | ADM3 只插入（刪除 0 行） | `--pause`：握手、凍結目標、base_hold decel／hold／零速、監督器、停止發布、`/wgmpc/pause_status`、`/door/pause_ack_solver` |
| `door_k3m_policy_p.py` | K3MPolicy 子類 | SETTLE ⇒ PAUSE、全程監看、結果收尾、總時長上限 25 s |
| `door_k3m_task_node_p.py` | 任務節點複本（原行改動 3 處：策略類別與 schema） | 暫停話題、凍結序號、觀察器就緒閘與心跳 |
| `door_k3m_check_p.py` | 判定器擴充 | K3-M 判準（PAUSE→END_HOLD）＋暫停 P1–P4 獨立重算 |
| `door_k3m_observer.py` | 觀察器複本 | `--door-max-deg` |
| `door_k3m_pause_sup.py`、`door_k3m_pause_mode.py`、`door_k3m_applied_src.py` | 新純函式 | 監督器、握手、套用來源 |
| `run_door_k3m_pause.sh` | 跑法複本 | 上列節點與參數、觀察器、凍結 |

增益、權重、限制、門檻、走廊與幾何監看都不改。

## 3. 測試（全部通過）
| 測試 | 項目 |
|---|---|
| `test_door_k3m_pause_sup.py` | 7：基準靜止 ⇒ 兩種停止都確認；**情境一 門依 lw_1 實錄續轉 ⇒ 不報停止**；**情境二 零命令底盤依 lw_1 實錄漂移 ⇒ 不報實際停止**；CONFIRM 中移動、停止後移動、缺樣本、觀察未滿、無套用回報逾時、窗口拒收 |
| `test_door_k3m_pause_mode.py` | 4：確認；執行端各種 nack（含中止優先於重送）；求解端只在已發布後回 ack、停止則丟棄；凍結目標不符、兩種逾時 |
| `test_door_k3m_applied_src.py` | 3（真 CmdChainE2＋K2AbortLatch）：延遲、重送、停更、非全身控制、中止優先、舊命令在停止發布後重新生效 |
| `test_dl3_wgmpc_wg2_adm4_pause_ros.py` | ROS 假世界 4 情境：P1 正常（ZERO_PUB 6.17 s、CONFIRM 6.22 s、STOPPED 6.27 s、DONE 16.3 s；STOPPED 後 0 列發布）；P2 門續轉 ⇒ timeout；P3 暫停中停止優先；P4 舊命令 ⇒ old_command_after_stop |
| ADM3 假世界測試改指 ADM4（不帶 --pause） | M1–M3 全過（無 --pause 時與 ADM3 同行為） |
| `test_door_k3m_policy_p.py` | 5：正常 PAUSE→DONE、目標保持、FAIL 中止、握手失敗、PAUSE 中非設計接觸中止（紀錄相位 PAUSE）、總時長逾時 |
| `test_door_k3m_task_node_p_ros.py` | 1：free 路由照舊；暫停請求、兩個 ack、狀態進策略 |
| 實測 | 觀察器退出 ⇒ 任務節點 1.03 s 後記 lw_observer_unhealthy 並結束 |
| `test_door_k3m_check_p.py` | 3：一致 ⇒ 無理由；12 種不一致各自理由（舊序號、非零套用、停止後發布、5 步不足、移動、覆蓋不足、缺步、凍結目標非最後已發布、ack 不一致、結果非 DONE、停止時刻不一致、展開路徑重疊）；PAUSE→END_HOLD 對應不改原紀錄 |
| 回歸 | `test_door_k3m_policy.py`、`test_door_k3m_pause.py` 通過 |

## 4. 首趟申請（下一步，待核可）
- **範圍**：單趟 `door_k3m_pause_1`，小角度（8°）協同開門＋受控暫停結束。
- **保護**：既有閂鎖與監看、獨立觀察器、92 °C；觀察上限＝STOPPED 後 10 s（監督器）＋暫停總時長 25 s；結束模擬不算實際停止。
- **結果分報**：K3-M 任務、命令停止、實際停止、保護完整性；附影片（該趟實錄重播 side／close）。
- **重播的前提**：ev2rp 的重播分支是以 ev2 場景建的；K3-M 執行端（ev4）的場景與 ev2 相同（同一門資產、機器人、停位），重播沿用 ev2rp。
