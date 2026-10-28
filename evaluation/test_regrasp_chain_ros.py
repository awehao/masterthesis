#!/usr/bin/env python3
"""換抓 ROS 聯測（Codex r65／r66；無 Isaac）：**真**任務節點 v10（--regrasp）＋**真** ADM19（bench9 參數＋W2＋--regrasp）＋**真**轉接器 k3m3（v3＋台架就緒）
＋**真**觀察器 door_k3m_observer ＋ 替身執行端 door_regrasp_fake_exec（三情境共用；受控體為名目運動學、接觸為合成並標記）。
情境（各自網域、各自結果目錄；不以一情境代替另一情境）：
  i   啟動短測：凍結 → 轉接器就緒 → 首解宣告 → v3 封裝經 W2Exec 進鏈 → 第 1 步套用確認（scenario start）；
      i-a 正常；i-b 轉接器晚 3 s 訂閱；i-c 首步前收到較新封裝（newer_before_step1）——首筆仍為宣告序號、較新者留在收件匣。
  iii 完整路徑：台架凍結 → INIT_CHECK → ENGAGE → OPEN → SETTLE → PAUSE／HELD → mode 1–6 → 短保持（scenario full、無故障）。
  ii  故障注入（full；HELD 之前流程照真節點走，不直接指定 HELD）：stop_spike、stop_gap、drop_c3_mode4、abort_at_commit6、lose_force_hold。
結果寫 TMP/<tag>/（fake room_run.json、contact_v3.jsonl、task.json、align_solver.json、各行程輸出）；摘要寫 results/door/regrasp_chain_ros_<tag>.json（不覆寫）。

    source /opt/ros/jazzy/setup.bash && source install/setup.bash && python3 evaluation/test_regrasp_chain_ros.py i-a [i-b i-c iii ii-stop_spike …]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TMP = os.environ.get('RGC_TMP', '/tmp/claude-1000/regrasp_chain_ros')
DOM0 = int(os.environ.get('RGC_DOMAIN', '120'))
PRM = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r11.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
REF = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v5.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
P = json.load(open(PRM))
TABLE = os.path.join(WS, P['protection_scope_change']['corridor_table'])
ENVL = P['protection_scope_change']['envelope']['cli']
OBS = str(P['protection_scope_change']['observer_door_max_deg'])
COORD = P['control_cost_change']['coord_topic']
CASES = {'i-a': ('start', 'none', 0.0), 'i-b': ('start', 'none', 3.0), 'i-c': ('start', 'newer_before_step1', 0.0), 'iii': ('full', 'none', 0.0)}
for _f in ('stop_spike', 'stop_gap', 'drop_c3_mode4', 'abort_at_commit6', 'lose_force_hold'):
    CASES[f'ii-{_f}'] = ('full', _f, 0.0)


def spawn(cmd, env, log):
    return subprocess.Popen(cmd, cwd=WS, env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT, text=True, start_new_session=True)


def run(tag, dom):
    scen, fault, ad_delay = CASES[tag]
    d = os.path.join(TMP, tag)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    run_id = f'rgc_{tag.replace("-", "_")}'
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom), PYTHONUNBUFFERED='1')
    py = sys.executable
    procs = {}
    procs['fake'] = spawn([py, '-u', os.path.join(HERE, 'door_regrasp_fake_exec.py'), '--params', PRM, '--run-id', run_id, '--regrasp-ref', REF,
                           '--corridor-table', TABLE, f'--envelope={ENVL}', '--scenario', scen, '--fault', fault, '--out', d], env, os.path.join(d, 'fake.log'))
    time.sleep(2.0)
    procs['obs'] = spawn([py, '-u', os.path.join(HERE, 'door_k3m_observer.py'), '--out', os.path.join(d, 'lw_observer.json'), '--door-max-deg', OBS],
                         env, os.path.join(d, 'obs.log'))
    procs['task'] = spawn([py, '-u', os.path.join(HERE, 'door_coopen_core_task_node10.py'), '--out', os.path.join(d, 'task.json'), '--run-id', run_id,
                           '--grasp-basis', GB, '--w1b-params', PRM, '--corridor-table', TABLE, '--regrasp', '--regrasp-ref', REF,
                           '--w2-mode-run-id', run_id], env, os.path.join(d, 'task.log'))
    procs['solver'] = spawn([py, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm20_node.py'), '--bench-start', '--bench-params', PRM, '--require-topic-target',
                             '--park-hold', '--offset-observer', 'interval', '--mode-switch-run-id', run_id, '--cmd-env', '--run-id', run_id, '--pause',
                             '--pause-arm-zero', '--pause-active-hold', '--pause-arm-win-s', '0.5', '--coord-topic', COORD, '--offset-hold-on-free',
                             '--offset-update-in-open', '--offset-moving', '--offset-free', '--continuous', '--arm-model', 'setpoint', '--arm-ident', IDENT,
                             '--cmd-topic', '/wgmpc/cmd_world', '--target-topic', '/drawer/tcp_target', '--target-offset', '0', '0', '0', '--N', '5',
                             '--rate', '20', '--delay-comp-state-cycles', '1.6', '--delay-comp-cmd-cycles', '1.0', '--near-target-gamma', '1.0',
                             '--w-s', '0.001', '--w-a', '0.05', '--w-s-arm', '0.05', '--margin-guard', '--reach-pos-m', '0.005', '--reach-rot-rad', '0.02',
                             '--hold-s', '2.0', '--duration-s', '900', '--duration-sim-s', '400', '--u-prev-policy', 'strict', '--assume-initial-rest',
                             '--w2-retreat', '--w2-hold-r9', '--w2-held', '--w2-held-max-s', '60', '--w2-mode-run-id', run_id, '--regrasp',
                             '--out', os.path.join(d, 'align_solver.json')], env, os.path.join(d, 'solver.log'))
    time.sleep(ad_delay)
    procs['adapter'] = spawn([py, '-u', os.path.join(HERE, 'arm_vel_adapter_k3m3.py'), '--in-topic', '/wgmpc/cmd_world', '--consumer-node', '/door_k2_sim',
                              '--cmd-env', '--cmd-env-v3', '--env-in-topic', '/wgmpc/cmd_env', '--bench-ready-run-id', run_id], env, os.path.join(d, 'adapter.log'))
    t_lim = 120 if scen == 'start' else 700
    w0 = time.monotonic()
    while time.monotonic() - w0 < t_lim and procs['fake'].poll() is None:
        time.sleep(1.0)
    time.sleep(3.0)
    for k in ('task', 'solver', 'adapter', 'obs', 'fake'):
        p = procs[k]
        if p.poll() is None:
            p.terminate()
    for k, p in procs.items():
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    ld = (lambda f: json.load(open(os.path.join(d, f))) if os.path.exists(os.path.join(d, f)) else None)
    return {'tag': tag, 'dir': d, 'run_id': run_id, 'fake': ld('room_run.json'), 'task': ld('task.json'), 'solver': ld('align_solver.json'),
            'rc': {k: p.returncode for k, p in procs.items()}, 'wall_s': time.monotonic() - w0}


def summarize(r):
    f, s, tk = r['fake'] or {}, r['solver'] or {}, r['task'] or {}
    b = f.get('bench') or {}
    ann = ((s.get('bench') or {}).get('announced')) or {}
    rg = (f.get('regrasp') or {}).get('core') or {}
    out = {'tag': r['tag'], 'rc': r['rc'], 'wall_s': r['wall_s'], 'announced_seq': ann.get('source_seq'),
           'released': (b.get('release') or {}).get('released'), 'first_intake': b.get('first_intake'),
           'first_step': {k: (b.get('first_step') or {}).get(k) for k in ('ok', 'why', 'applied_source_seq', 'step', 'sim_t')},
           'adapter_ready': (s.get('bench') or {}).get('adapter_ready'), 'faults': f.get('faults'), 't_end': f.get('t_end'), 't_mode': f.get('t_mode'),
           'k2_latch': f.get('k2_latch'), 'regrasp_abort': f.get('regrasp_abort'), 'core_abort': rg.get('abort'),
           'auth_commits': [(c.get('mode_seq'), c.get('step'), c.get('applied_confirmed')) for c in ((f.get('w2') or {}).get('auth') or {}).get('commits', [])],
           'auth_rejections': [x.get('why') for x in ((f.get('w2') or {}).get('auth') or {}).get('rejections', [])][:10],
           'gates_ended': rg.get('ended'), 'hold': {k: (rg.get('hold') or {}).get(k) for k in ('state', 'why', 't')},
           'gate2': {k: ((rg.get('gates') or {}).get('gate2') or {}).get(k) for k in ('state', 'why', 't_zero', 't_cmd', 't')},
           'handover': rg.get('handover'), 'task_phase': tk.get('final_phase'), 'task_abort': tk.get('abort'),
           'task_phases': [e.get('to') for e in tk.get('policy_events', []) if 'to' in e][:30], 'door_final_deg': f.get('door_final_deg'), 'base_final': f.get('base_final'), 'synthetic': f.get('synthetic')}
    return out


def main():
    tags = sys.argv[1:] or ['i-a']
    res = {}
    for i, tag in enumerate(tags):
        print(f'[{tag}] 開始', flush=True)
        r = run(tag, i)
        sm = summarize(r)
        res[tag] = sm
        p = os.path.join(HERE, 'results', 'door', f'regrasp_chain_ros_{tag}.json')
        if os.path.exists(p):
            p = p.replace('.json', f'_{int(time.time())}.json')
        json.dump(sm, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
        print(json.dumps(sm, ensure_ascii=False, default=float)[:1500], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
