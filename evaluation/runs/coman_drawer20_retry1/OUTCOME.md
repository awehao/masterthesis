# coman_drawer20_retry1：**環境不合格 ＋ 程式中止，未進入任務**

**不是操作失敗** —— 任務從未開始。三題（完整操作、拉動期間同動、執行期時效）
**都沒有證據**，不得以本趟填任何一題。

| 項目 | 值 |
|---|---|
| 時間 | 2026-09-30 14:02 起（UTC 06:02） |
| RUN_ID / domain | `coman_drawer20_retry1` / 71 |
| 起動前檢查 | **0 項失敗** |
| F1（固定關節守衛） | **通過** —— `world→根 固定關節 0 個（模式 free_base 期望 0 個）` |
| 中止點 | 進入物理主迴圈後立刻 `NameError` |
| 進入任務 | **否**（未等到 `/clock`，`exit 3`） |
| 趟次資料 | 無 |
| CPU 峰值 | 58.0 °C（限 92 °C） |
| 殘留程序 | 無，cleanup 完成 |

## 兩個原因

### 1 程式：`NameError: name 'MACHINE_WIRED' is not defined`（已修）

```
File "evaluation/isaac_coman_drawer_sim.py", line 1343, in main
    print(f'[coman] 正式釋放資格來源：狀態機（目前 MACHINE_WIRED={MACHINE_WIRED}）',
```

變數名是 `machine_wired`（第 1257 行）。**只是一行 log**，卻讓整趟中止。

這是 F1 之後**第三個**只存在於 `--free-base` ＋ `--machine` 路徑的缺陷 ——
該路徑從未真正執行過（29 趟歷史全是 `importer_fix_base`）。
已用 pyflakes 掃過協同鏈七個檔案的「未定義名稱／使用前參照」，**全部無**，
同類問題應已一次抓完。

### 2 環境：**GPU 不可用**，本趟即使跑起來也不該採信

| 項目 | 首趟 11:48 | 本趟 14:02 |
|---|---|---|
| 核心 | 7.0.0-**31** | 7.0.0-**34** |
| `cudaErrorNoDevice` | **0 次** | **131 次** |

* `nvidia-smi`：`couldn't communicate with the NVIDIA driver`
* `/dev/nvidia*`：不存在
* 真正的 `nvidia` 核心模組未載入（只有無關的 `nvidia_wmi_ec_backlight`）
* `nvidia-driver-595-open` 已安裝，但 `dkms status` 為空
  ⇒ 模組未為 7.0.0-34 編出

機器在兩趟之間重開進新核心，NVIDIA 模組沒跟上。Isaac 退為 CPU-only：
PhysX GPU 功能跳過、`updateDeformableTransforms` 略過、無 GPU 算繪。

**三題中有兩題（動態同動、執行期時效）直接取決於物理與 RTF**，
CPU-only 的數字不能代表這個配置的表現。因此本趟**不採信**，不填任何一題。

## 重試前置條件

GPU 必須恢復（`nvidia-smi` 正常、`/dev/nvidia0` 存在）才重跑。
本目錄保留原樣；重試另開目錄。
