"""核心段任務節點 v8 ＋ K3-B 任務節點 r5（祖先 w1b2）三方合併 → v10，再插入 RG10（Codex r63／r64）。重跑：python3 results/door/mk_task_node10_patch.py [--check]
合併：git merge-file -p door_coopen_core_task_node8.py door_k3m_task_node_w1b2.py door_k3m_task_node_k3b5.py ⇒ 3 處衝突：
  (1) 檔頭：兩段並存；(2) 策略建構：保留 v8 的包裝鏈；k3b5 的 W2／K3-B 釋放策略分支**不採**（v10 的 W2 只用於 --regrasp；未給 --regrasp 卻給
  --w2-mode-run-id／--w2-release／--w2-unload ⇒ 啟動拒絕）；(3) 紀錄欄位：兩段並存。
RG10 只插入（每個錨點恰好一次；另移除不採分支的 5 個 K3-B 策略匯入）：參數、匯入、換抓策略建構、v3 接觸摘要訂閱、換抓相位以策略 coord 取代 W1b 引導、換抓報告。
"""
import subprocess
import sys

m = subprocess.run(['git', 'merge-file', '-p', 'door_coopen_core_task_node8.py', 'door_k3m_task_node_w1b2.py', 'door_k3m_task_node_k3b5.py'],
                   capture_output=True, text=True)
assert m.returncode == 3, m.returncode                               # 恰 3 處衝突
s = m.stdout


def take(s):
    a = s.index('<<<<<<< ')
    b = s.index('\n=======\n', a)
    c = s.index('>>>>>>> ', b)
    return s[:a], s[s.index('\n', a) + 1:b + 1], s[b + 9:c], s[s.index('\n', c) + 1:]


HDR = ('# 核心段任務節點 v10（Codex r63／r64）：v8（台架開門線）與 k3b5（W2／K3-B 線）以 w1b2 為祖先三方合併；衝突 3 處（檔頭、策略建構、紀錄欄位）。\n'
       '#   策略：--regrasp 時 BG.bench_grasp2(IG2.init_grasp2(AG2.gated2(door_regrasp_policy6.RegraspPolicy6)))——開門前綴的台架初態、INIT_CHECK、\n'
       '#   接近閘控與 v2 監看照 v8；PAUSE → HELD（W2 mode 1 測試保持）→ 換抓 mode 2–6（policy6：v3 監看器獨立、mode 5 預抓取時鐘、張爪前銜接預檢、\n'
       '#   登錄核驗）；未給 --regrasp ⇒ 與 v8 同行為；給了 W2／K3-B 旗標卻沒給 --regrasp ⇒ 拒絕（k3b5 的釋放策略分支不採）。\n'
       '#   RG10：v3 接觸窗口摘要 /door/contact_v3_summary → pol.on_c3；換抓相位（RG*）以策略輸出 coord 取代 W1b phase_coord（後者在 RG 相位為關閉）；\n'
       '#   task.json 另記 regrasp（策略 rg_report）。\n'
       '# ---- 以下為 v8 線說明 ----\n')
pre, ours, theirs, rest = take(s)
s = pre + HDR + ours + '# ---- 以下為 k3b5 線說明 ----\n' + theirs + rest
pre, ours, theirs, rest = take(s)
POL = ours.replace(
    "    pol = BG.bench_grasp2(IG2.init_grasp2(AG2.gated2(PP2.K3MP2Policy)))(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()), a_H=a_H)   # C1／G2\n",
    "    if (a.w2_mode_run_id or a.w2_release or a.w2_unload) and not a.regrasp:   # RG10：v10 的 W2 只用於換抓\n"
    "        raise SystemExit('任務節點 v10：--w2-mode-run-id／--w2-release／--w2-unload 只在 --regrasp 下使用 ⇒ 拒跑')\n"
    "    if a.regrasp:                                                          # RG10：換抓策略（台架開門前綴的包裝鏈不變）\n"
    "        if not a.w2_mode_run_id or not a.regrasp_ref:\n"
    "            raise SystemExit('--regrasp 需 --w2-mode-run-id 與 --regrasp-ref ⇒ 拒跑')\n"
    "        if a.w2_release or a.w2_unload:\n"
    "            raise SystemExit('--regrasp 與 --w2-release／--w2-unload 互斥 ⇒ 拒跑')\n"
    "        rg_pkg = json.load(open(a.regrasp_ref))\n"
    "        _okr, _whyr = REGY.validate(rg_pkg)\n"
    "        if not _okr:\n"
    "            raise SystemExit(f'換抓參考包登錄核驗不過：{_whyr} ⇒ 拒跑')\n"
    "        rg_fn = GM3.backend_v3(self_mode=a.regrasp_geom_self_mode)\n"
    "        pol = BG.bench_grasp2(IG2.init_grasp2(AG2.gated2(RP7.RegraspPolicy7)))(\n"
    "            cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()), a_H=a_H,\n"
    "            w2_run_id=a.w2_mode_run_id, req_delay_s=a.w2_req_delay_s, commit_timeout_s=a.w2_commit_timeout_s,\n"
    "            test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s,\n"
    "            ref_pkg=rg_pkg, kin=WholeBodyKinematics.from_urdf_file(a.urdf), geom_v3_fn=rg_fn)\n"
    "    else:\n"
    "        pol = BG.bench_grasp2(IG2.init_grasp2(AG2.gated2(PP2.K3MP2Policy)))(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()), a_H=a_H)   # C1／G2\n")
assert POL != ours
s = pre + POL + rest
pre, ours, theirs, rest = take(s)
s = pre + ours + theirs + rest
assert '<<<<<<<' not in s and '>>>>>>>' not in s
for _m in ('k3b', 'k3b2', 'k3b3', 'k3b4', 'k3b5'):                        # 不採的 K3-B 釋放策略分支所用匯入（RG10 移除）
    _ln = [x for x in s.splitlines(keepends=True) if x.startswith(f'import door_k3m_policy_{_m} as ')]
    assert len(_ln) == 1, _m
    s = s.replace(_ln[0], '')


def ia(anchor, text):
    global s
    assert s.count(anchor) == 1, anchor
    s = s.replace(anchor, anchor + text)


def ib(anchor, text):
    global s
    assert s.count(anchor) == 1, anchor
    s = s.replace(anchor, text + anchor)


ia("    ap.add_argument('--w2-unload', action='store_true')                          # K3-B r4：需 --w2-release\n",
   "    ap.add_argument('--regrasp', action='store_true')                            # RG10：換抓 mode 2–6（需 --w2-mode-run-id、--regrasp-ref）\n"
   "    ap.add_argument('--regrasp-ref', default='')                                 # RG10：已登錄參考包\n"
   "    ap.add_argument('--regrasp-geom-self-mode', default='full', choices=('full', 'sphere'))   # RG10：v3 監看自碰模式（離線核對用 full）\n")
ia("import door_bench_start2 as BENCH                                         # noqa: E402  E1／F2：台架契約 r2\n",
   "import door_regrasp_policy7 as RP7                                        # noqa: E402  RG10：換抓策略 r7（r6＋mode 3 前等求解端 release_settled）\n"
   "import door_k3m_geom_monitor_v3 as GM3                                    # noqa: E402  RG10：v3 監看後端（雙把手）\n"
   "import door_regrasp_ref_registry as REGY                                  # noqa: E402  RG10：參考包登錄核驗\n")
ia("        nd.create_subscription(String, '/door/w2_mode_state', _w2_state_cb, 50)\n",
   "        if a.regrasp:                                                      # RG10：執行端 v3 接觸窗口摘要\n"
   "            nd.c3_bad = 0\n"
   "            nd.c3_n = 0\n"
   "\n"
   "            def _c3_cb(m_):\n"
   "                try:\n"
   "                    pol.on_c3(json.loads(m_.data), nd._t_now())\n"
   "                    nd.c3_n += 1\n"
   "                except Exception:                                          # noqa: BLE001\n"
   "                    nd.c3_bad += 1\n"
   "            nd.create_subscription(String, '/door/contact_v3_summary', _c3_cb, 50)\n")
ib("            nd.coord_pub.publish(Float64MultiArray(data=[float(x) for x in _coord]))\n",
   "            if a.regrasp and str(pol.phase).startswith('RG') and act.get('coord') is not None:   # RG10：換抓段由策略給底盤引導\n"
   "                _coord = list(act['coord'])\n"
   "                rep.setdefault('regrasp_coord', {'n': 0})['n'] += 1\n")
ia("            rep['w2'] = dict(pol.w2_report(), n_bad_state=getattr(nd, 'w2_bad_state', None))\n",
   "        if a.regrasp:                                                          # RG10\n"
   "            rep['regrasp'] = dict(pol.rg_report(), n_c3=getattr(nd, 'c3_n', None), n_c3_bad=getattr(nd, 'c3_bad', None),\n"
   "                                  ref=os.path.relpath(a.regrasp_ref, WS), ref_sha256=hashlib.sha256(open(a.regrasp_ref, 'rb').read()).hexdigest())\n")
if '--check' in sys.argv:
    cur = open('door_coopen_core_task_node10.py').read()
    print('identical' if cur == s else 'DIFFERENT')
    sys.exit(0 if cur == s else 1)
open('door_coopen_core_task_node10.py', 'w').write(s)
print('written')
