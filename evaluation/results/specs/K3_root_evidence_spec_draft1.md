# 最小取證規格 draft-1：根與門扇的逐步狀態三點讀取（只加紀錄、不改控制；**只送規格，未實作、未開跑**）

依 Codex 20261023_233900 第 2 節。

**目的**：補上 `K3_lw_contract_check_note_r2.md` 第 2 節列為未知的層，回答：
1. 門扇姿態與角速度的讀取分支（PhysX tensor 或 USD 回退），以及兩者在同一步的關係；
2. 根速度寫入的實際參數、寫入後回讀值，以及物理步後的根位姿與速度；
3. 零命令期間，物理步後的根速度是否非零、量值多少。

**不回答**：原因、門速保護是否保守或失效、K3-M 停止。

## 1. 配置
- 沿用 `run_door_k3ab_lw.sh`／lw r3 的全部參數、保護與觀察器。控制、門檻、增益都不改。
- 唯一差異：執行端改用版本化複本 `isaac_door_k2_sim_ev2r.py`，由 ev2 **只插入**取證碼，另寫 `root_evidence.jsonl`。
- 取證碼的例外不得中斷模擬，比照 EV 的 `KEV.safe`：錯誤計數，該欄記 null 並附原因。
- **只跑一次**，結果照實保留、不補跑。仍須事前核可。

## 2. 每物理步記錄（`root_evidence.jsonl`，一步一行）
| 欄位 | 讀取時點 | 內容 |
|---|---|---|
| `step`、`t` | — | 與 room steps 同一個步號 |
| `handle` | 每步 | `leaf_v.is_physics_handle_valid()`；機器人 articulation view 的 handle 有效性；`SimulationManager.get_physics_sim_view() is not None` |
| `root_pre` | A：`ex.step` 之後、API 寫入之前 | `robot.get_world_pose()`、`get_linear_velocity()`、`get_angular_velocity()` |
| `api_args` | 寫入當下 | 實際傳給 `set_linear_velocity`／`set_angular_velocity` 的 float32 向量（原樣），以及所用的 yaw |
| `root_post_write` | B：兩個 setter 之後、`apply_action` 之前 | 根速度回讀（線、角） |
| `leaf_pre_step` | C：記錄列的讀取點（寫入後、`world.step` 前；與 room 列相同） | 門扇姿態（四元數、位置）與角速度三軸，**另附讀取分支**（tensor 或 USD；以 handle 判定） |
| `root_post_step`、`leaf_post_step` | D：`world.step()` 之後立即 | 根位姿與速度；門扇姿態與角速度三軸 |
| `timing_s` | — | 取證碼耗時（每步） |

- D 與下一步的 A 之間沒有其他寫入，所以預期相同；這一點本身也要核對（`D_k` vs `A_{k+1}`）。
- 既有的關節反力（evidence 的 j6 wrench、dof efforts）照舊，不新增感測。

## 3. 只讀分析（取證後；`door_k3ab_root_evidence_analysis.py`，之後另寫）
- **handle**：全程有效／無效的步數，與分支。
- **根速度**：
  - `api_args` vs `root_post_write`：寫入是否原樣生效；
  - `root_post_write` vs `root_post_step`：物理步改變了多少；
  - 零命令段（閂鎖後）`root_post_step` 的分佈；
  - `root_post_step` 速度 × dt 與 D 點位姿差分的一致性。
- **門扇**：
  - C 與 D 點角速度 vs 姿態差分（同區間積分比較，沿用 r2 方法）；
  - 三軸角速度，看 ω_x、ω_y 是否非零；
  - tensor 分支下，速度與姿態是否一致。
- **核對**：`D_k` vs `A_{k+1}` 的差。
- **耗時**：取證耗時對 RTF 的影響。若顯著，照實報告，結果標為「取證改變了時序」。
- 只描述；不做根因主張，也不改保護。

## 4. 開跑前交付（另送審）
- ev2r 與 ev2 的差異表（只插入）、跑法複本 `run_door_k3ab_lw_ev2r.sh`。
- 取證寫入的例外測試；無 Isaac 的語法與匯入核對。
- 新凍結（lw r3 清單＋新檔）。

## 5. 不宣稱
- 單趟取證；不是停止驗收；不解除 K3-M 停止進入條件；5 mm 門檻不變。
