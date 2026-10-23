# 只讀契約核對 r2 ＋ API 語意查核 ＋ 門速候選離線比較（Codex 20261023_233900）

r1（`K3_lw_contract_check_note.md`、`door_k3ab_lw_contract_check.json`）保留不改。本文件取代 r1 的數字與三處措辭。

## 1. 契約核對 r2（`door_k3ab_lw_contract_check_r2.py` → `results/door/door_k3ab_lw_contract_check_r2.json`）
**窗口來源**：逐趟相位事件（task.json policy_events 的 HOLD、OPEN、策略中止，以及 room events 的 k2_abort_latched），來源寫入輸出。
| 趟 | HOLD | OPEN | 策略中止 | 閂鎖 |
|---|---|---|---|---|
| lw_1 | 18.72 s | 19.76 s | 21.48 s | 21.51 s（步 2149） |
| ev_1 | 19.40 s | 20.42 s | 22.16 s | 22.19 s（步 2217） |

r1 把固定的 18–19.5 s 稱為 HOLD，那段含 ENGAGE；ev_1 的 OPEN 起點 r1 誤用 20.5 s。**以上兩處撤回。**

### 底盤（本體座標，mm/s、mrad/s；命令＝base_cmd_body，位姿差分＝meas_vb_body）
| 趟 | 段 | 命令 | 位姿差分 | 零命令步 |
|---|---|---|---|---|
| lw_1 | HOLD [18.72, 19.76) | (+3.36, −2.01, −3.01) | (−0.08, +0.01, +0.10) | 0/104 |
| lw_1 | OPEN→中止 | (+2.86, −1.91, −6.54) | (+0.31, −0.07, +1.93) | 0/172 |
| lw_1 | 閂鎖後 1 s–窗末 | (0, 0, 0) | (−2.58, +1.35, −1.82) | 911/911 |
| ev_1 | HOLD [19.40, 20.42) | (+3.36, −2.01, −2.92) | (−0.08, 0.00, +0.08) | 0/102 |
| ev_1 | OPEN→中止 | (+2.86, −1.90, −6.40) | (+0.31, −0.07, +1.80) | 0/174 |
| ev_1 | 閂鎖後 1 s–窗末 | (0, 0, 0) | (−2.56, +1.27, −1.60) | 206/206 |

- HOLD 命令線速 3.9 mm/s，位姿差分約 0.08 mm/s。
- 閂鎖後位姿差分與 HOLD 命令線速的方向餘弦為 −0.998（lw_1）、−0.997（ev_1）。
- 主要觀察維持；只是紀錄層的對應，不確認原因。

### 門角與門速：同區間積分比較
- 對同一區間 [k_a, k_b]，比較角度增量 Δq 與紀錄角速度的數值積分（左端、右端、梯形），以及時移 s＝−2…+2 步的左端積分。下表為整段的「積分 ÷ Δq」。
- 另以 0.1 s 子區間計算相對殘差，中位數見 JSON。

| 趟 | 段 | Δq（rad） | 左 | 右 | 梯形 | s−2 | s−1 | s+1 | s+2 |
|---|---|---|---|---|---|---|---|---|---|
| lw_1 | OPEN→中止 | 0.01091 | 2.36 | 2.37 | 2.37 | 2.33 | 2.34 | 2.37 | 2.39 |
| lw_1 | 閂鎖後 0–1 s | 0.00890 | 1.36 | 1.34 | 1.35 | 1.39 | 1.38 | 1.34 | 1.33 |
| lw_1 | 閂鎖後 1–3 s | 0.00760 | 4.47 | 4.47 | 4.47 | 4.44 | 4.44 | 4.47 | 4.49 |
| lw_1 | 閂鎖後 3 s–窗末 | 0.02678 | 4.69 | 4.69 | 4.69 | 4.69 | 4.69 | 4.69 | 4.69 |
| ev_1 | OPEN→中止 | 0.01109 | 2.33 | 2.35 | 2.34 | 2.30 | 2.31 | 2.35 | 2.37 |
| ev_1 | 閂鎖後 0–1 s | 0.00839 | 1.31 | 1.31 | 1.31 | 1.33 | 1.32 | 1.31 | 1.30 |
| ev_1 | 閂鎖後 1–3 s | 0.00787 | 3.92 | 3.93 | 3.93 | 3.89 | 3.90 | 3.93 | 3.97 |

- 0.1 s 子區間相對殘差中位數：OPEN 約 +1.2；閂鎖後 0–1 s 為 +0.26～+0.45；1 s 之後約 +3.4～+3.6。
  - 左端、右端、梯形與 ±1 步時移之間的差異 ≤ 0.1。
- **描述**：在所列區間，紀錄角速度的積分是角度增量的 1.3–4.7 倍。換用左端／右端／梯形或 ±2 步時移，比值變化 ≤ 0.08。
  - 因此**在 ±2 步時移範圍內**，這個幅值差不能以時移解釋。
  - 原因未查；更大時移或其他時間語意不在本次範圍。
- **措辭修正**：
  - 刪除 r1 的「本列角速度對應下一個 world.step；同列角與速指向不同物理步區間」，改為：在所選窗口，紀錄角速度與前向一步角度差分的相關較高；尚不能據此確認 API 的時間語意或延遲。
  - 刪除 r1 的「長窗平均不受一步時移影響，所以不是時序造成」，改為上段的積分比較描述。
  - 刪除 r1 的「方向上會較早觸發」，保留：兩種量的幅值不一致，現有結果不足以判定保護保守或失效。

## 2. API 語意查核（只讀程式與本機安裝版；未讀執行期設定）
**依啟動路徑與本機安裝檔查核**（Codex 20261023_234952 修正措辭）：啟動路徑 `ISAAC_PY=~/venvs/isaacsim-6.0.1/bin/python`，本機安裝 isaacsim 6.0.1.0。歷史趟未保存執行期模組來源與 handle 狀態，**不補寫成已驗證**。執行端 import 的 `isaacsim.core.prims`（`RigidPrim`、`SingleArticulation`）與 `isaacsim.core.api.World`，來源在 `extsDeprecated/isaacsim.core.prims` 與 `extsDeprecated/isaacsim.core.api`。

| 項目 | 程式語意 | 本趟狀態 |
|---|---|---|
| 門扇姿態 `RigidPrim.get_world_poses()` | handle 有效 ⇒ `_physics_view.get_transforms()`（PhysX tensor）；無效 ⇒ USD／xform 回退路徑 | **未知**：handle 有效性沒有紀錄 |
| 門扇角速度 `get_angular_velocities()` | handle 有效 ⇒ `_physics_view.get_velocities()[:, 3:6]`（rad/s）；無效 ⇒ 讀 USD `angularVelocity` 屬性（deg/s），再 deg2rad | 同上，**未知** |
| handle 有效條件 | `SimulationManager.get_physics_sim_view() is not None and self._physics_view is not None`；`_physics_view` 在 `initialize()` 時由 `create_rigid_body_view` 建立 | ev2 在 `world.reset()` 之後呼叫 `leaf_v.initialize()`，**推定有效但未驗證** |
| 根速度寫入 `SingleArticulation.set_linear_velocity`／`set_angular_velocity` | → `Articulation.set_linear_velocities`：handle 有效時先 `get_root_velocities()`，覆寫線速度（或角速度）分量，再 `set_root_velocities()`；兩次呼叫各寫一部分。座標：世界座標根連桿速度（是否為質心速度依 PhysX tensor 語意，本次未查證） | handle 有效性與回讀值都沒有紀錄 |
| `set_angular_velocity([0, 0, wz])` | 同時把 x、y 角速度寫成 0 | 程式可證 |
| `world.step(render=False)` | → `SimulationContext.step` → `PhysicsContext._step(current_time, update_fabric=False)` → `_physics_sim_interface.simulate(dt, current_time)`；不更新 fabric | `simulate` 是否含 fetch results、子步數，**未查證** |
| 物理設定 | ev2 未設定求解器類型、迭代數、子步、CCD、穩定化；physics_dt＝rendering_dt＝0.01 s | 執行期實際值（場景預設、URDF 匯入預設）**未知**，沒有保存 |
| 讀取時序 | 每步：讀位姿 → ex.step → 寫根速度 → 手臂 apply_action → 讀門角門速與根位姿並記錄 → world.step | 同列的門角、門速與根位姿都是**寫入後、步進前**的讀值 |

- **推論界線**：若 handle 有效，姿態與速度同出自 PhysX tensor view，不能直接歸因於「USD 姿態與 PhysX 速度不同步」。
- 若 handle 無效則走 USD 回退路徑；`update_fabric=False` 本身**不能證明** USD 未更新，只列為待核對的回退路徑。兩種情況目前都不能排除，需要取證。

## 3. 門速候選離線比較（`door_rate_candidates_offline.py` → `results/door/door_rate_candidates_offline.json`）
- **資料**：8 趟（k3a_1、adm_1、int_1、ev_1、au_ev_1、ax_ev_1、ab_ev_1、ab_lw_1）。
- **候選**：raw（現用）與因果後向差分 bd1／bd5／bd10（0.01／0.05／0.1 s；只用過去樣本，窗口內缺步或非有限 ⇒ 無值，不補零）。
- **缺步**：8 趟皆 0。初始化時，bdN 開頭 N 步無值；raw 從第一步就有值。
- **峰值（|值| 最大，rad/s）**：

| 段 | raw | bd1 | bd5 | bd10 |
|---|---|---|---|---|
| OPEN→策略中止 | 0.0235–0.0327 | 0.0118–0.0280 | 0.0102–0.0204 | 0.0098–0.0184 |
| 閂鎖後 0–1 s | 0.0297–0.0337 | 0.0256–0.0312 | 0.0192–0.0230 | 0.0181–0.0201 |
| 閂鎖後 1 s–窗末 | 0.0243–0.0463 | 0.0141–0.0232 | 0.0099–0.0130 | 0.0064–0.0100 |

- **超過 0.05 rad/s**（原 door_abort_rate）：8 趟所有段、所有候選都是 0 步，沒有首次觸發時刻可比較。
- **描述**：
  - 每段的 raw 峰值都 ≥ 各候選峰值；
  - 差分窗口越長，峰值越低（平滑）；
  - bd10 有 10 步初始化空窗，並對缺步敏感。
- **界線**：
  - 線上任務節點以 50 Hz 收狀態，約每 2 物理步一筆，線上差分的間距與本表不同；
  - 前向差分需要未來樣本，線上不可用；
  - 既有紀錄無缺步，缺步行為只依定義描述；
  - **不據此替換或並行加入保護**。

## 4. 不變
lw_1 的停止結論不變；5 mm 門檻不變；K3-M 停止進入條件未解除；不核可新模擬。最小取證規格見 `K3_root_evidence_spec_draft1.md`（只送規格，不開跑）。
