# K3 停止後長窗隔離診斷（單趟；規格 draft-1；未實作、未開跑，須 Codex 事前核可）

依 Codex 20261023_225057 第 3 節，下一步選 (ii)。

**要回答的問題**：固定底盤 K3-A-B 配置中，執行端閂鎖之後：
- 門與底盤**何時停住**，或在 10 s 內有沒有停住；
- 停住時，以及觀察窗內，殼—橫桿等近似距離與接觸的實際值。

**不回答**：K3-M 底盤可動情況下的停止；暫停控制律；開門能否完成。

## 1. 配置（沿用 K3-A-B EV，不改控制）
| 項目 | door_k3ab_ev_1（既有） | 本診斷 door_k3ab_lw_1 | 理由 |
|---|---|---|---|
| 執行端 | isaac_door_k2_sim_ev2.py（含 evidence） | 同 | — |
| 求解器、展開、轉接 | 同 run_door_k3ab_ev.sh 的參數 | 同（增益、權重、限制全不改） | — |
| 任務節點 | door_k3ab_task_node.py | `door_k3ab_lw_task_node.py`：只新增 `--post-abort-observe-s`，預設 3.0（與原值相同） | 原本 K2Config.post_abort_observe_s = 3.0 s（模擬時間）後就 finish，所以實錄只到閂鎖後約 3.1 s |
| 中止後觀察 | 3.0 s（從策略中止起算） | **10.0 s**（同一參數，從策略中止起算；策略→閂鎖約 0.03 s，所以閂鎖後約 9.97 s） | 已知 3 s 不夠 |
| 牆鐘中止後上限 wall_post_abort_s | 60 s | 60 s（不改） | 10 s 模擬時間在現有 RTF 下約 20 s 牆鐘 |
| 暫停控制、B／U／X 以外的新介入 | 無 | 無 | — |
| 跑法 | run_door_k3ab_ev.sh | `run_door_k3ab_lw.sh`（複本：只換任務節點、加 `--post-abort-observe-s 10`、換凍結清單） | — |
| 判定器 | door_k3ab_check.py | 同一判定器照常判 K3-A-B（預期仍是未完成）；另加只讀分析 `door_k3ab_lw_analysis.py`（§3） | 不改判通過 |

- **只跑一次**。未出現 finger_contact_lost（例如到 DONE 或其他中止）時**不強迫製造失接觸**、不補跑；照實分析當趟的停止。
  - 若到 DONE，求解器在 DONE 停止，分析從 DONE 步起算，並標明不是閂鎖停止。

## 2. 記錄（既有欄位已涵蓋，不新增控制路徑）
- 每個物理步（0.01 s）的 room `steps`：
  - 命令：base_cmd_body、applied_arm；
  - 量測：meas_vb_body、base_xyth、q_arm_meas；
  - 門：door_q_rad、door_qdot_rad_s；
  - 手指：q_finger、finger_contact_n；
  - 接觸：leaf_contacts、wall_contacts；
  - 腕：wrist_force_n。
- evidence.jsonl 逐步（既有 EV 趟為 2524／2524 步齊全）：arm_sp、j6_reaction_wrench、接觸與摩擦。
- 時間點：
  - 策略中止時刻：task.json policy_events；
  - 執行端閂鎖步：room events `k2_abort_latched`，本診斷的時間零點。

## 3. 「停住」的事前定義（只讀分析）
- 以 0.1 s（10 物理步）位移差分估速度，不用逐步 meas_vb_body。逐步值雜訊約 1–3 mm/s，與門檻同量級。
- 四個分量**同時**滿足以下條件、且**連續 1.0 s** ⇒ 停住；停住時刻＝這 1.0 s 的起點：
  - 底盤：|Δp|/0.1 ≤ 1 mm/s，且 |Δψ|/0.1 ≤ 0.005 rad/s；
  - 門：|Δq|/0.1 ≤ 0.002 rad/s；
  - 實測手臂：max |Δq_arm|/0.1 ≤ 1e-3 rad/s。
- 數值與受控暫停的完成判準相同。
- 另外分別報告各分量**各自**首次連續 1.0 s 滿足的時刻。
- 閂鎖後 10 s 內未停住 ⇒ 報**未停止**，附窗末值（門角、底盤位移與偏航、末 1 s 速率）。**不自動延長、不補跑。**
- 幾何：沿實測軌跡逐步（不是每 2 步）重算近似距離，報取樣最小值；手指開度取兩指平均（近似）。
- **只讀觀察**：閂鎖後殼—橫桿（非手指類）< 5 mm 的區段，逐段保存起訖時間、最小值與當時的接觸。
  - 這是離線分析，不重設中止、不恢復操作、不改判通過，也不稱為監看觸發：策略中止後已離開 ACTIVE_MONITOR。
- 分兩段報告：
  - 段 A：策略決定 → 執行端閂鎖；
  - 段 B：閂鎖後。

## 4. 實驗保護與提前結束（與控制成功無關）
既有保護，不改：
- run_guarded_v2.sh 的 92 °C 熱中止；
- 執行端 --sim-limit 240 s；
- 任務節點牆鐘總上限與中止後牆鐘上限 60 s；
- 各節點 wait_or_stop 期限；
- 收尾只停本趟 PID。

新增候選（請裁定是否需要）：任務節點在閂鎖後的觀察期內，若符合下列任一條件，就提早 finish 並記 `experiment_guard`：
- 實測門角 ≥ 5°（約為中止時門角的 7 倍）；
- 任一非設計接觸 ≥ 5 N，連續 0.1 s。

說明：
- 這兩項只是實驗保護，結束模擬不代表控制器成功停住。
- 觸發時報「觀察提前結束」，停止判定記為「未驗證」。
- 數值只是事前候選，不依結果調整。

## 5. 版本與凍結
- 新檔：
  - door_k3ab_lw_task_node.py：由 door_k3ab_task_node.py 只插入參數；
  - run_door_k3ab_lw.sh；
  - door_k3ab_lw_analysis.py ＋ 反例測試；
  - 凍結 `freeze_door_k3ab_lw_dev_r1.sha256`：K3-A-B 凍結清單 ＋ 上述新檔。
- 既有 K3-A-B 檔案與實錄都不改。

## 6. 不宣稱
- 單趟介入觀察，不做根因主張。
- 停住或未停住都只代表這個配置的這一趟。
- 不是 K3-M 停止的證據，也不解除 K3-M 實跑的進入條件。
