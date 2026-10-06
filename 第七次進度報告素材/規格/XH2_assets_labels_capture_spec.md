# XH2：資產與標註程式＋train／dev 採集——子包規格（draft-1，送審；核可前不採集、不訓練）

2026-10-06。依封存登錄 `XH1_registration.yaml`（`freeze_xh1_registration.sha256`）。**只渲染 train／dev 資產**（R 8 個＋N 2 個 × 2 條路徑）；test 資產（R_x25_150、R_x40_300、N_k40）在模型／門檻／後端規則凍結前不產生資產檔以外的任何東西（資產檔與路徑檔可產生並記雜湊，但不渲染）。凍結檔（`drawer_asset.py`、`isaac_drawer_room_sim.py`、`run_wrist_v0.sh`、`dl0_autolabel.py`、`dl0_paths.py`）不改，一律另建版本。

## 1　資產
- **R（drawer_unit/1）**：由 `drawer_unit_bar26.yaml` 複製，只改 `handle.bar.radius = d/2`、`handle.bar.length = L`、支柱 `post_l／post_r` 中心 x = ∓(L/2 − 0.015)、尺寸 (0.016, 0.040, d − 0.002)、支柱中心 y、z 與 bar26 相同（櫃體本地 (±x, −0.265, 0.550)）；
  其餘（櫃體、抽屜本體、關節、物理參數、`grasp_surface` 舊欄位照 bar26 原樣——標註與評估不讀這些欄位）。檔名 `drawer_unit_xh_<id>.yaml`，標 `xh_asset_id`。
- **N（drawer_unit/2，新 schema）**：/1 全部欄位，但 `handle` 改為可選，新增 `distractors: [{type: knob_cylinder_y, center_local, diameter, protrusion}]`：
  圓鈕為沿本地 −y（朝外）的圓柱，**根部貼在前板外面**（y = −0.245），突出量 p 由前板外面起算 ⇒ 圓柱軸段 y ∈ [−0.245 − p, −0.245]、中心 x = 0、z = 0.550（與 bar26 橫桿中心同高）；直徑與突出量照登錄（30／28、35／30、40／32 mm）。
  固定於抽屜本體（隨抽屜運動），碰撞形狀為解析圓柱。
- **建構程式**：`drawer_asset_v2.py`——對 /1 規格的 `shapes_world`／`build_usd` 與凍結 `drawer_asset.py` 逐項相同（測試：同一 /1 檔兩版 `shapes_world` 輸出逐值相等；`build_usd` 由程式碼 diff 限定只新增 /2 分支）；/2 加上圓鈕。
- **擷取模擬器**：`isaac_drawer_room_sim_xh.py`＝凍結模擬器複本，唯一差異 `import drawer_asset_v2 as DA`；擷取腳本 `run_wrist_xh.sh`＝`run_wrist_v0.sh` 複本，唯一差異 `--drawer-asset "$DRAWER_ASSET"`（必設）與模擬器檔名。相機、取樣率、裁切與 DL0 完全相同。

## 2　觀測路徑（確定性；採集前產生並記雜湊）
共同：5 Hz；相機觀看目標點＝**觀測參考中心**（R：橫桿中心；N：圓鈕外端面中心＝(0, −0.245 − p, 0.550) 櫃體本地 → 世界）；影像 x 方向提示＝世界 +x；滾轉 0；
底盤在相機後方，前距在 {0.25, 0.30, 0.35, 0.40, 0.45, 0.50} m 中取 IK 位置誤差最小者（同 `dl0_paths`）；IK、關節內縮 0.05 rad、逐物理步線性內插皆沿用 `dl0_paths`（複本 `xh_paths.py`，只改路徑定義與參考中心）。
相位 φ = 2π·(seed mod 1000)/1000（seed＝登錄的資產種子）。`smooth(a, b, s)` 與 `dl0_paths` 相同（s 夾在 [0, 1] 的平滑步進）。
- **A_front**（81 格）：距離 d(i) = smooth(3.0, 0.4, i/60)（i ≤ 60），之後 smooth(0.4, 0.2, (i − 60)/20)；
  相機 = 參考中心 + (0.04·sin(2π·0.1·t_i + φ), −d(i), 0.03)，t_i = 0.2 i s；全程朝參考中心（無朝外段）。
- **B_oblique**（66 格）：方位單位向量 u = (sin 30°, −cos 30°, 0)（相機在 +x 側前方）；距離 d(i) = 2.0（i < 15），之後 smooth(2.0, 0.3, (i − 15)/50)；
  相機 = 參考中心 + d(i)·u + (0, 0, 0.06)；觀看目標 = smooth(朝外點, 參考中心, (i − 10)/5)，朝外點＝相機 + 1.5·(0.6, 0.8, −0.1)（前 10 格朝外＝負例影格，10–15 格轉向）。
- 每資產兩條路徑檔 `results/vision/xh_paths/<id>_<template>.json`＋登錄 `XH2_paths_registration.yaml`（雜湊、IK 誤差、FK 預估類別）；IK 不合格（位置 > 2 mm 或姿態 > 0.5°）格數照列，**不因結果換路徑**。

## 3　標註（`dl0_autolabel_v2.py`，凍結 `dl0_autolabel.py` 複本）
- 射線一律 **PC2 像素中心 +0.5**（與辨識反投影同一契約）。
- R：目標＝真值橫桿圓柱（半徑、長度、中心讀資產檔）可見像素 × 量測深度一致（沿用 DL0 規則與容差）；支柱＝背景；ignore＝外圈 1 px、端蓋（只限 R 橫桿）、無效深度、背面、近裁切。
- N：圓鈕側面與外端面可見像素＝背景，另存 `distractor` 遮罩（同一射線法，含外端面圓盤）；不 ignore。
- 影格類別：positive（有效影格中有可評分目標）／negative（無目標，含只見 N、朝外）／excluded（有 R 但目標像素全被 ignore）。
- 一致性測試：v2 對 bar26 既有 DL0 開發影格產生的標籤與 v1 的差異只來自 +0.5 契約（報差異像素數與位置分佈）；N 遮罩在合成影格上與解析幾何一致。

## 4　採集
- train：R_bar26、R_t22_160、R_t24_280、R_t30_240、R_t34_180、R_t38_220、N_k35；dev：R_d28_200、R_d32_260、N_k30 ⇒ 10 資產 × 2 路徑 = 20 次只渲染擷取（`run_guarded.sh`＋`RUNNER=evaluation/run_wrist_xh.sh`，92 °C 熱中止不變）。
- 採集前凍結清單：資產檔、建構程式、模擬器複本、擷取腳本、路徑檔、標註程式、登錄檔。採集後只做技術完整性檢查（影格數、時間匹配、雜湊）再標註。
- 不渲染 test 資產；不依辨識結果更換路徑或資產。

## 5　待審
1. N 圓鈕幾何（根部貼前板外面、同高、直徑／突出量）與 /2 schema 範圍。
2. 路徑定義（A 無朝外段、B 有 10 格朝外負例）是否足以提供負例；或 A 也加朝外段。
3. 以複本隔離凍結檔的做法與一致性測試是否足夠。

---

# draft-2（依 Codex XH2 審查；以本節為準）
1. **無橫桿支援**（不是只換匯入）：`isaac_drawer_room_sim_xh.py` 在資產無橫桿時只允許 `--wrist-v0` 擷取（其他模式仍 rc 22）、把手 prim 為 None、改以圓鈕 prim 作觀測參考；
   `wrist_xh_capture.py`：R 照舊記橫桿真值；N 記圓鈕幾何中心、朝外軸與**觀測參考中心（外端面中心）**，**不寫 handle_center 欄位**；truth.json 標 `target_kind`（R_bar／none）。不建假橫桿、不把圓鈕冒充橫桿、不沿用 bar26 真值。
   圓柱幾何中心 y = −0.245 − p/2（櫃體本地），外端面中心 y = −0.245 − p 才是路徑參考點。
2. **路徑登錄照實**：前距選擇＝先最少 IK 不合格格數、再最小最大位置誤差、同分取候選順序較前者（`dl0_paths.make_seq` 原規則）；
   分開保存設計姿態數（81／66）、物理步數（(設計姿態數 − 1) × 20，內插不含最後端點）、擷取嘗試、新影格與拒絕數，不靜默少算分母。`xh_paths.py` 匯入凍結 `dl0_paths` 只設定其模組變數。
3. **標註一致性基準**：凍結 `dl0_autolabel.py` 原始碼**只把射線像素座標改 +0.5** 後執行，與 v2（bar26 參數）逐像素比較 mask、ignore 與計數；尺寸與 N 參數由資產讀取，缺欄即拋出。
4. A 不加朝外段；實際負樣本數由標註判定，不保證 B 前 10 格全為負例。
5. 採集時先跑排程內 R、N 各一條核對通路，正常再完成其餘；不另加功能趟，不因辨識結果換路徑。
