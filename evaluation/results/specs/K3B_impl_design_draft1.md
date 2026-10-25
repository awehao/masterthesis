# K3-B 受控釋放實作設計 draft-1（接線設計＋已完成的純邏輯核心；**尚未接入控制、不模擬**）

依 Codex 20261025_230812：draft-2 修訂與離線核對完成後，可進入版本化控制實作及離線／假世界測試，一包送審。
- 本文件先交代接線設計，並附上已完成、已測的純邏輯核心。
- 控制接線涉及三個控制架構選擇（§2 的 D1、D2、D4）。這三點直接決定求解端與執行端怎麼改，所以實作前先請裁定。
- 判準規格見 `K3B_release_clear_spec_draft2.md`。

## 1. 已完成（純邏輯＋測試，未接入任何節點）
| 檔案 | 內容 | 測試 |
|---|---|---|
| `door_k3b_release.py` | ReleaseMonitor：狀態 IDLE→A_OPENING→B_CONFIRMED→C_CLEAR→C_RESTORED→RETREAT_DONE→STOW_READY／FAILED；判準取自規格 draft-2，常數 REG 固定 | `test_door_k3b_release.py` 11 項 |
| `door_k3b_pair_geom.py` | 兩指定配對的距離與預測，與幾何監看同一預測規則；每指用實測開度 | `test_door_k3b_pair_geom.py` 3 項 |

**ReleaseMonitor 的判準內容**：
- 單調門角帶，含 T_pred 0.2 s 預測；
- 四軸工具偏移，相對 Trans(s_cmd·a_H)·E_rel；
- 三段配對語意；
- 段 A 18 N 上限；
- 確認：95 %＋0.1 N＋0.5 s、期限 1.5 s；
- 無效資料規則：缺欄、非有限、過期、缺步、n_invalid 都不算解除，連續 0.2 s 判失敗；
- 退出條件：s ≥ 0.02、配對 d 與 d_pred 都 ≥ 5 mm；期限 2 s；
- 5 N 恢復准許：靜窗 0.5 s，須先有請求；提前或未請求即讀回判失敗；
- 收臂前置 Δ ∈ [−3, +2.5]°；
- 一般保護：非手指／自碰 5 mm、底盤 20 mm 含預測、錨點 20 mm／2°、腕力 30 N、非設計接觸 0.1 N。

**實作途中修正的核心缺陷**：缺步核對原本對照「最後有效步」，無效段結束後會永遠判缺步。已改為對照「最後看到的步」，由 K3 測試固定。

**配對距離核對**：HELD 實測組態算出 −0.19／−0.36 mm，與離線 v4c 一致（G1）。

## 2. 接線設計與待裁定的選擇
**D1 模式序列（沿用 W2 的權威、候選驗證、逐步確認與判定器框架）**
| mode_seq | 名稱 | 命令形狀 | 監看 profile | 手指目標 |
|---|---|---|---|---|
| 1 | RELEASE_PREP_TEST | 保持形 | pause_hold／release_prep_test（既有） | 閉 |
| 2 | RELEASE_OPEN | **保持形**（手臂 0、底盤錨點伺服） | `release_open` | **開**：執行端只在 mode 2 提交步才放行張爪命令 |
| 3 | RETREAT | 手臂追退離目標、底盤錨點伺服 | `release_retreat` | 開 |

- mode 2 提交步就是 t_open（規格 R1：mode_seq、profile、手指目標同一步切換）。
- mode 3 候選驗證不能用保持形，改為：
  - 有限；
  - 底盤保持形；
  - 手臂速率 ≤ 既有上限；
  - 相對實際套用 u_prev 的加速度框（物理步 dt）；
  - 候選內容為退離首筆。
- Codex 已指出雙流只適用於測試模式。所以 mode 3 **不用**「候選＝保持命令」的雙流假設：
  - 求解端在 mode 3 準備期間，一般流維持 mode 2（保持）；
  - 旁路候選是「以退離目標求解的首筆」，獨立 source_seq；
  - 提交後一般流改為退離。
- 這需要求解端 ADM12（ADM11 只插入）：mode 3 提交後，監督器由 HELD 交給「底盤保持＋手臂追 /drawer/tcp_target」，即既有 PARK_HOLD 追蹤路徑。

**D2 收臂是否納入本包**
- 收臂路線（v4c：退離終點→Q_PRE_DOOR→K2_READY）是關節空間軌跡，目前只有展開節點有同一律的實作。
- 若納入，需要 mode 4 STOW 與展開節點反向相位，或求解端關節目標模式。
- **提議本包止於 RETREAT_DONE＋收臂前置核對（STOW_READY），之後停在退離終點主動保持（HELD 同一機制）**；收臂另包。

**D3 監看放在哪一端**
- ReleaseMonitor 在**任務端**以 20 Hz 執行，與幾何監看相同。
  - 接觸資料用 /door/contacts 窗口最大值：窗口涵蓋每個物理步，取最大值保守。
  - 門速用同步流 0.1 s 後向差分（SETTLE 已用的訊號），不用 tensor 門速。
- 任一失敗 ⇒ 任務端中止 ⇒ /door/k2_abort ⇒ 執行端 K2 閂鎖（既有失效路徑，規格 §5）。
- 執行端不跑 100 Hz 配對幾何，以免拖慢 RTF（幾何監看單輪約 15 ms）。

**D4 5 N 恢復的權威**
- 任務端在 ReleaseMonitor.restore_allowed 為真時送 `/door/finger_effort_req`。
- **執行端自行再核**：
  - 已在 mode 3；
  - 最近 0.5 s 逐物理步兩指—橫桿接觸 ≤ 0.1 N；
  - 開度 ≥ 95 %。
- 通過才在物理步設 5 N 並讀回（EV5 同一機制），回 ack；任務端以 ack 的讀回結果餵 effort_readback_ok。
- 執行端拒絕 ⇒ 任務端中止。

**D5 判定器**
- 以 room／task／solver 紀錄重算規格 R1–R8，包括：
  - 執行端 mode 2／3 提交步、手指目標切換步、mode 3 首筆實際套用（C0–C11 擴充）；
  - 任務端 ReleaseMonitor 事件與執行端逐步資料交叉核對：t_conf 由 room 逐步重算，不只信任務端；
  - 不另造新 DONE 語意：DONE＝STOW_READY 後保持 ≥ 3 s。

## 3. 假世界與測試計畫（實作包）
- 純邏輯：
  - ModeAuthority5（mode 2／3 各自的候選驗證）；
  - 手指命令閘門；
  - 5 N 握手權威。
- 真 E2 鏈接線：mode 3 首筆限幅、拒收、延遲；mode 2 張爪與 mode_seq 同步。
- ROS 假世界：ADM12＋adapter_w2＋替身端點＋W2Exec 執行端角色。情境：
  - 正常：mode 1→2→3、退離 0.10 m；
  - mode 3 候選被拒；
  - 退離中來源中斷；
  - 恢復請求被執行端拒絕。
- 真任務節點路由：父類開門與暫停用替身，同 W2c。
- 判定器反例：合成資料，同 W2c 做法。

## 4. 請裁定
1. D1：mode 2 保持形＋張爪閘門、mode 3 退離候選的驗證內容、ADM12 的交接方式。
2. D2：本包止於 RETREAT_DONE＋STOW_READY，收臂另包。
3. D3：監看在任務端 20 Hz（窗口最大值、0.1 s 差分門速）。
4. D4：5 N 恢復由執行端以逐步資料自行再核後執行。
5. 規格 draft-2 §9 的四項候選值（18 N、T_pred 0.2 s、收臂 [−3, +2.5]°、更嚴的近區帶）。
