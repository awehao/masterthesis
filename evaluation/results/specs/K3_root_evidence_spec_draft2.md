# 最小取證規格 draft-2：根與門扇逐步三點讀取＋影片產生方案（已實作與離線測試；**未開跑，須核可**）

依 Codex 20261023_234952 修正 draft-1（`K3_root_evidence_spec_draft1.md`，不改）。這輪完成後收斂到「取證結果＋協同開門」，不再擴大診斷支線。

## 1. 差異表（相對 lw r3：run_door_k3ab_lw.sh／door_k3ab_lw_1）
| 項目 | LW r3 | LW-EV2R | 性質 |
|---|---|---|---|
| 執行端 | isaac_door_k2_sim_ev2.py | `isaac_door_k2_sim_ev2r.py`：ev2 的**只插入**複本（diff 刪除 0 行、新增 153 行；pyflakes 警告與 ev2 相同） | 只加紀錄與重播分支 |
| 新旗標 | — | `--root-evidence-out`、`--physics-snapshot-out`（空 ⇒ 不讀任何新 API）；`--replay`（ev2 原本有參數但未實作；此處實作為只渲染分支） | — |
| 任務節點、觀察器、求解器、轉接、展開、所有參數、門檻 | 同 | 同 | 不改 |
| 跑法 | run_door_k3ab_lw.sh | `run_door_k3ab_lw_ev2r.sh`：只換執行端並加兩個取證旗標、換凍結 | 複本 |
| 新純函式 | — | `door_root_evidence.py`（讀到即複製、例外與非有限記原因、快照三分類、模組來源）；寫檔沿用 `door_k3_evidence.EvidenceWriter` | 新增 |
| 分析器 | — | `door_k3ab_root_evidence_analysis.py` | 新增 |
| 影片 | — | `run_door_replay.sh`＋`door_replay_video.py` | 新增（只渲染） |
| 凍結 | lw r3 | `freeze_door_k3ab_lw_ev2r_dev_r1.sha256`＝lw r3 清單＋新檔 | — |

## 2. 取證契約（依 Codex 第 2 節補齊）
- **時間與資料所有權**：
  - A、B、C 標本步時間 t；D 另記 `D_t`＝t＋physics_dt，為步進後時間。
  - 所有 API 回傳值在讀取當下由 `REV.grab` 轉成新建的 Python float 串列，不保存 view 或張量。
- **讀取點**（每物理步一列 `root_evidence.jsonl`）：

| 點 | 位置 | 內容 |
|---|---|---|
| A | `ex.step` 後、API 寫入前 | `handle_leaf_A`、`handle_robot_A`；根位姿 `A_root_pre`（位置 3＋四元數 wxyz 4）；根速度 `A_root_vel_pre`（線 3＋角 3）；門扇 `A_leaf_pre`（位姿 7＋速度 6） |
| API | 兩個 setter 呼叫的實際參數 | `api_args`（float32 原樣：線速度 3＋角速度 3，世界座標）、`api_yaw`、`cmd_body` |
| B | 兩個 setter 之後、apply_action 前 | `B_root_vel_post_write`、`handle_robot_B` |
| C | 記錄列讀取點（寫入後、world.step 前） | `handle_leaf_C`；`C_leaf_pre_step`；`C_root_pre_step`（位姿＋速度）；`C_recorded`（執行端 door_q_qd 的 q、q̇） |
| D | world.step 之後立即 | `D_t`、`handle_leaf_D`、`handle_robot_D`；`D_root_post_step`；`D_leaf_post_step` |
| — | — | `evidence_s`（本步取證耗時）、`missing`（缺欄）、`errors`（原因） |

- **讀取分支**：各點記 `is_physics_handle_valid()`，只稱「依有效性判定的預期分支」，不是分支的直接證據。
- **一致性比較**（分析器）：
  - 速度積分對位姿差分，使用同一世界座標、同一剛體；
  - 根速度參考點（根連桿原點或質心）未釐清前，兩者的差**不稱為物理失配**；
  - `D_k` 對 `A_{k+1}` 量化差值，不先假定相等。
- **完整性**：列數對 room 步數、缺步、重複步、缺欄或錯誤列、寫檔器 complete、快照與模組來源，任一不成立 ⇒ `evidence_complete=False`，診斷不判完整；控制照原流程結束。
  - 非有限值記 None 與 'nonfinite'（寫檔 allow_nan=False，不會被 NaN 擊穿）。
- **物理設定快照**（啟動時一次）：
  - **USD 宣告值**：PhysicsScene、門扇、鉸鏈關節、機器人 articulation root 的 authored 屬性原樣字串。
  - **執行期回讀值**：physics_dt、rendering_dt、求解器類型、CCD、穩定化、broadphase、GPU dynamics、裝置、機器人求解迭代數、自碰設定、啟動時 handle 有效性。
  - **未知**：讀不到或未宣告者列出原因，不以預設值代替。
  - 鉸鏈關節狀態不在 articulation 內，不為此新增 articulation ⇒ 列為未知。
  - 另記執行期模組來源：RigidPrim、SingleArticulation、World、articulation view 與門扇 physics view 類別的原始檔路徑與 sha256，以及套件版本。

## 3. 測試（離線，無 Isaac）
- `test_door_root_evidence.py` 8 項：立即複製（來源改動不影響已存值）、例外與非有限記原因、bool 不當數值、tuple 位姿、缺欄清單、快照三分類、模組來源與 sha256、寫檔器對 NaN 的處置。
- `test_door_k3ab_root_evidence_analysis.py` 4 項（合成資料）：
  - 完整一致時 evidence_complete，api、B、D_k／A_{k+1} 差皆 0，零命令速度與位姿差分比為 1，門扇 −ω_z 積分比為 1；
  - 缺步、重複、缺欄、寫檔未完成、缺快照 ⇒ 不完整並列原因；
  - handle False／None 計數、api 參數不符可偵測；
  - 門扇角速度 4 倍不一致可量化為 4.0。
- ev2r：ast 解析通過；與 ev2 的 pyflakes 警告相同（無新增未定義名稱）。實際 Isaac 啟動只能在核可的實跑中驗證。

## 4. 實跑申請（單趟）
- `RUN_ID=door_k3ab_lw_ev2r_1`，由 run_guarded_v2.sh 包 92 °C 熱中止。只跑一次，不論結果如何都不補跑；任務 ABORTED 照實保留。
- **不即時錄影**：即時渲染會讓 RTF 由約 1.0 降到約 0.5，改變控制時序。影片改用該趟實錄重播（§5）。
- 結果分報：任務判定、觀察器保護完整性、取證完整性、實際觀察長度與停止分析、三點讀取分析。
- 不做根因主張；不改門速保護；5 mm 門檻不變；不解除 K3-M 停止進入條件。

## 5. 影片產生方案（Howard 要求；成功與失敗都保留）
- **方式**：該趟實錄重播（render-only）。`run_door_replay.sh` 以 `isaac_door_k2_sim_ev2r.py --replay` 建同一場景，逐幀設底盤位姿、手臂與手指關節、門扇繞鉸鏈轉到紀錄門角。
  - 每幀速度歸零、推一個物理步讓變換生效後擷取，同 isaac_drawer_room_sim 的既有重播做法。
  - 不起 ROS、不跑控制、不改任何既有實錄。來源 room_run.json 的 sha256 記入 run.log。
- **視角**：side（側後方全景）與 close（夾爪與把手近景），各一支；10 Hz 模擬時間 → 20 fps（2 倍速）。
- **標註**（`door_replay_video.py`）：
  - 頂欄：RUN_ID、「該趟實錄重播（非即時錄影；2 倍速）」、模擬時間、相位、門角、閂鎖後底盤位移；
  - 關鍵時刻橫幅：對準開始、夾持開始、夾持保持、OPEN 開始、策略中止（失接觸）、執行端閂鎖、停止後漂移；
  - 另輸出 `.keymoments.json` 供 README 引用。
  - 已用合成灰底幀與 lw_1 的時間軸驗證標註與編碼流程。
- **重播需要啟動模擬器**，範圍一併申請：
  1. `door_k3ab_lw_1` 的 side 與 close 重播（優先補充）；
  2. `door_k3ab_lw_ev2r_1`（若核可並跑完）的 side 與 close 重播。
  - 每支各一次，熱中止同上；只渲染。
  - 若首幀取景明顯不當（看不到夾爪或門），只調相機參數重渲染一次，並記錄調整；不改其他任何東西。
- **交付**：`第七次進度報告素材/影片/`＋`README.md`，列 RUN_ID、目的、結果、影片連結、關鍵時刻與「該趟實錄重播」標示。
- **重播的界線**：每幀推一個物理步，接觸可能造成單步微小位移，但每幀位姿重設、不累積；影片只供展示，不作量測依據。
