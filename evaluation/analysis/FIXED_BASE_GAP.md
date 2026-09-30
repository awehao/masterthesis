# F 組（固定底盤）對照的實作缺口（**唯讀盤點，本輪不實作**）

依規劃 §4.1：F 組必須同時做到
(a) 物理上固定底盤、(b) **在 QP 內**加入底盤三分量速度為零的約束、
(c) 與 M 組共用手臂控制、目標生成、安全配置及正式判準。

## 1. 已具備

| 需求 | 現況 | 位置 |
|---|---|---|
| **QP 內**的底盤零約束 | **已存在** `SafetyConfig.fix_base`；為真時 `_box_rows` 把 `vmax[:3] = 0` ⇒ `−0 ≤ v₀..v₂ ≤ 0` **進入約束集** | `wholebody_safety_filter.py`（`fix_base` 欄位、`_box_rows`） |
| 「不得求解後截掉」的理由已寫明 | 原註解：底盤能動時求解器可用「後退底盤」滿足屏障，事後歸零則手臂份額已按「底盤會幫忙」決定 | 同上 `fix_base` 上方註解 |
| 安全層可設定 | ROS 參數 `fix_base`（預設 False）→ `SafetyConfig(fix_base=…)` | `wholebody_safety_node.py` |
| 求解端會吃同一份設定 | `_box_rows(self.cfg, …)` ⇒ 設 `self.cfg.fix_base = True` 即生效 | `coman_pull_solver_node.py` `_constraints` |
| 物理固定底盤 | `import_urdf(ROBOT, fix_base=not a.free_base)`；模式守衛已改為模式相依（F 期望 1 個 world→根固定關節） | `isaac_coman_drawer_sim.py` |
| 共用手臂控制／目標生成／安全配置／正式判準 | 同一 runner、同一 v1／C1／R1.1／S1、同一 `PullTarget`／`PullTaskPolicy`／`HandoverMachine` | `run_coman_drawer20.sh` |

**結論：核心機制已存在，缺的是「接上」與「核對」。**

## 2. 缺少

| 代號 | 缺口 | 後果 |
|---|---|---|
| **B1** | 求解端**沒有 CLI 旗標**設 `cfg.fix_base` | F 組跑起來時求解器仍以為底盤可動 ⇒ 不符 §4.1 (b) |
| **B2** | runner **沒有 F 模式**。F 需要三處同時成立：執行端不帶 `--free-base`、安全節點 `fix_base:=true`、求解端 `--fix-base` | 只設其中一兩處會造成兩端不一致（求解器算有底盤、實體卻固定，或反之） |
| **B3** | **沒有一致性檢查**。入口核對的 F 段目前只檢查開放底盤那條路 | 三處不一致不會被擋下 |
| **B4** | 趟次輸出的 `base_fixation` 只記**物理**模式，**沒有記求解端是否 fix_base** | 趟後無法核對 §4.1 (b) 是否真的成立 |
| **B5** | **同動門檻不在任何已核准規格中**（見 `FIELD_INVENTORY` G7） | F／M 比較的關鍵指標無門檻可依；分析程式只能輸出 `cannot_determine` |
| **B6** | 底盤實測速度未記錄（G3） | F 組理論上底盤不動，但**無法用實測證明**；M 組更需要 |

## 3. 不得當成 F 組的東西

* 舊固定底座 **IK** 趟次（如 `drawer_220102_offset20`）**不是同方法對照** ——
  v1 `comparison.reference_differences_to_state` 已明載：仍有一次性交接偏置
  `b0`、控制方式與全身 QP 不同，**不得說成「只改了底盤是否固定」**。
* 自由底盤只送零速度**不是**物理固定（規劃 §4.1 已述）。
* 求解後把底盤命令截成零 —— `fix_base` 註解已明確拒絕。

## 4. 最小修改方案（**待確認後才實作**）

| 代號 | 修改 | 範圍 |
|---|---|---|
| **P1** | 求解端加 `--fix-base` 旗標 → `self.cfg.fix_base = True`，並寫入其輸出的 `args` | `coman_pull_solver_node.py`，約 3 行 |
| **P2** | runner 加 `BASE_MODE=free\|fixed`：`fixed` 時去掉執行端 `--free-base`、安全節點加 `-p fix_base:=true`、求解端加 `--fix-base` | `run_coman_drawer20.sh`，約 8 行 |
| **P3** | 起動前檢查加一條：`BASE_MODE` 與上述三處**必須一致**，不一致即具名失敗 | 同上的 preflight 區塊，約 6 行 |
| **P4** | 執行端把求解端的 `fix_base` 記進 `base_fixation`（由求解端輸出或 runner 寫入趟次目錄的 `run_config.json`） | 約 5 行；或改由 runner 落盤，不動執行端 |
| **P5** | 記錄底盤實測位置與速度（G3）—— 對 F／M 兩組都必要 | 執行端 `log` 加 `base_x/base_y/base_yaw/base_vx/base_vy/base_wz` 六欄，約 6 行；**會動 `LOG_COLS`，需同步欄名檢查** |
| **P6** | 同動門檻**另立比較規格**（不改 v1）：`base_lin_min_mps`／`joint_rate_min_rps`／`min_continuous_s`／`max_gap_s` | 新規格檔；由指導審定，**不由程式硬編** |

P1–P4 是「接上與核對」，不改控制器、增益或門檻。
**P5 會改變趟次輸出格式**（欄位新增），須一併更新欄名一致性檢查。
**P6 是規格決定，不是實作** —— 沒有它，F／M 比較的同動一欄只能是 `cannot_determine`。

## 5. 本輪未做

未實作上述任一項；未改控制器、增益、門檻或已核准規格；未開模擬器。
