# door_wide_w1f_60_2（W1f2 60° 驗證；凍結 w1f2 r4）——**環境失敗，任務未開始**

- 14:54:52 起跑：起跑前凍結核對通過、osqp 正常、熱守護 39 °C。
- 執行端 Isaac 在 180 s 內未建好場景。sim.log 只有 22 行，停在 `omni.kit.renderer.init` startup。
- 運行器 rc 83（場景未建起 ⇒ 中止），只清本趟 PID；起跑後凍結核對通過。
- 任務節點、求解端都沒有啟動，**沒有 task.json／room_run.json，無任務結果可判**。判定器 w1f2 因缺 task.json 無法執行，不是判定不通過。

## 當時主機狀態（只描述）
- GPU：RTX 5060 Laptop，445 MiB、0%、38 °C，無計算程序（只有 gnome-remote-desktop-daemon）。
- **Xorg（PID 238204）處於 D 狀態**；使用者層 systemd（PID 238520，約 14:50:09 起）、PID 1 systemd、一個 kworker 也處於 D 狀態。
  - `ps` 讀 /proc 卡住，加 timeout 也無法結束。
- 系統負載約 5；本趟 sim 程序已結束。
- 與 memory「amdgpu 畫面凍結（NVIDIA 算圖 → AMD 輸出的 fence 卡死）」的樣態相容，但沒有 dmesg 權限，**未確認原因**。

## 處置
- **不補跑、不重試**：主機顯示堆疊與 kernel 層異常未排除前，不再啟動 Isaac。
- 本趟目錄保留，作為環境失敗紀錄（比照 p2b_1／p2b_2）。
- 待 Howard 處理主機（需重開機或檢查），並經 Codex 確認後，以新 RUN_ID 重跑；凍結 r4 不變。
