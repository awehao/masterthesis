# K3-M 暫停 p4b：手臂速度改用 0.5 s 窗口（Howard 選定；實作＋測試＋凍結 p4b r1）

## 0. 決策
- Howard 2026-10-24 在三個選項中選定「**手臂改用 0.5 s 窗口**」。另兩個選項是：維持 0.1 s 並封存；先查微抖來源。
- 手臂停止門檻 **1e-3 rad/s 不變**，只把手臂速度的量測窗口由 0.1 s 改為 0.5 s。底盤與門仍用 0.1 s。
- 報告須寫明「手臂速度以 0.5 s 窗口計算」。

## 1. 依據（door_k3m_pause_p4_1，STOPPED 後 1.1 s；只描述）
- 手臂零命令下原地微抖：各關節範圍 ≤ 0.3 mrad，正負號來回 37–77 次，淨位移 ≤ 0.14 mrad。
- 手臂最大關節速度依窗口長度：

  | 窗口 | > 1e-3 的比例 | p95 | 最大 |
  |---|---|---|---|
  | 0.1 s | 28% | 0.0017 | 0.0023 |
  | 0.2 s | 1% | 0.0008 | 0.0011 |
  | 0.5 s | 0% | 0.0003 | 0.0005 |

- 0.5 s 窗口仍能抓到持續移動：測試 B2／J2 以 2e-3 rad/s 等速移動都判失敗。

## 2. 實作
| 檔 | 內容 |
|---|---|
| `door_k3m_pause_sup_p4b.py` | `WindowSpeedP4b`：底盤與門沿用 0.1 s；手臂以 0.5 s 窗口重算，有效性規則相同（步號遞增、間隔 ≤ 35 ms、不跨斷點、新鮮）。`ActiveHoldSupervisorP4b`：outcome 記 `arm_window_s`／`base_door_window_s` |
| `dl3_wgmpc_wg2_adm8_node.py` | ADM7 只插入（原行改動 1 處：監督器選擇加分支）：`--pause-arm-win-s`（需 `--pause-active-hold`） |
| `door_k3m_check_p4b.py` | check_p4 複本：P3 的手臂改 50 步窗口，其餘相同 |
| `run_door_k3m_pause_p4b.sh` | p4 跑法複本：ADM8＋`--pause-arm-win-s 0.5`、判定器 p4b |

## 3. 測試（系統 Python，全過）
| 測試 | 項目 |
|---|---|
| `test_door_k3m_pause_sup_p4b.py` 5 | 微抖 ±0.1 mrad／7 Hz：p4 失敗、p4b DONE；持續 2e-3 ⇒ timeout，停止後才動 ⇒ moved_after_stop；蠕變 0.8e-3 ⇒ DONE；停止後底盤漂移仍判失敗；0.5 s 窗口資料不足或跨斷點 ⇒ 無值；範圍檢查 |
| `test_dl3_wgmpc_wg2_adm8_pause_ros.py` P1–P5 | 同 ADM7 各項，另核 outcome 標 0.5 s，以及旗標相依拒跑 |
| `test_door_k3m_check_p4b.py` 5 | p4 各反例＋微抖（p4 判 moved、p4b 通過）＋持續移動（p4b 仍判 moved） |
| 本趟資料 | check_p4b 對 p4_1：命令停止 True；第一筆移動在 34.88 s（閂鎖後底盤漂移），手臂不再被判移動 |

## 4. 凍結與單趟
- 凍結 `freeze_door_k3m_pause_p4b_dev_r1.sha256`：p4 r1 的 193 檔＋本版 7 檔。
- 單趟 `door_k3m_pause_p4b_1`：乾淨環境、92 °C、失敗不補跑；附 side／close 實錄重播。
