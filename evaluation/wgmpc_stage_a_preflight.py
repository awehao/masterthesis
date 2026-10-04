#!/usr/bin/env python3
"""階段 A 的入口核對（**不啟動 Isaac、不重跑靜止基線**）。

只核受本輪影響的入口：門檻檔與配置指紋、目標與姿態、joint_margin 一致、
模式、距離節點參數的產生、交棒設定的自洽、必要檔案存在。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

THRESH = os.path.join(
    WS, 'evaluation/runs/wgmpc_stage_a_baseline_long120_020143',
    'drawer_threshold.json')
ASSET = os.path.join(WS, 'src/my_omnibot_description/config/drawer_unit.yaml')
STAGE_CFG = os.path.join(WS, 'src/ammr_wholebody_mpc/config/wgmpc_stage_a.yaml')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    rep, bad = {}, []

    def ck(name, fn):
        try:
            rep[name] = fn()
        except Exception as e:
            rep[name] = f'FAIL: {e}'
            bad.append(f'{name}: {e}')

    import drawer_asset as DA
    import wgmpc_stage_a_entry as E
    from wgmpc_wg2_node import validate_target_rot
    from ammr_wholebody_mpc.wgmpc_core import WGMPCConfig
    from ammr_wholebody_mpc.wgmpc_handover import HandoverConfig
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics

    spec = DA.load(ASSET)
    sc = yaml.safe_load(open(STAGE_CFG))
    pose = tuple(float(v) for v in sc['scene']['pose_xy'])
    backoff = float(sc['pre_contact_pause']['approach_backoff_m'])

    def _thresh():
        b = json.load(open(THRESH))
        if b.get('verdict') == 'rejected':
            raise RuntimeError('門檻檔是被拒絕的基線')
        fp = b['config_fingerprint']
        # 指紋要與**本輪會用的**資產與擺放一致
        import hashlib
        sha = hashlib.sha256(open(ASSET, 'rb').read()).hexdigest()
        if fp['asset_sha256'] != sha:
            raise RuntimeError(f'資產雜湊不符：門檻 {fp["asset_sha256"][:12]}… '
                               f'vs 現在 {sha[:12]}…')
        if [float(x) for x in fp['drawer_pose_xy']] != list(pose):
            raise RuntimeError(f'擺放不符：門檻 {fp["drawer_pose_xy"]} vs '
                               f'階段 A 設定 {list(pose)}')
        return {'file': os.path.relpath(THRESH, WS),
                'opening_tol_m': b['opening_threshold']['threshold'],
                'contact_tol_n': b['contact_force_threshold']['threshold'],
                'fingerprint_matches_asset_and_pose': True}
    ck('threshold', _thresh)

    def _target():
        p, R, meta = E.compute_pre_contact_pause(spec, pose, 0.0, backoff,
                                                 source='preflight')
        E.check_target(p, R, spec, pose, 0.0, backoff)
        validate_target_rot(R.reshape(-1))
        want = [float(v) for v in sc['pre_contact_pause']['tcp_world']]
        if max(abs(float(x) - y) for x, y in zip(p, want)) > 1e-4:
            raise RuntimeError(f'目標 {np.round(p,5).tolist()} 與設定檔 {want} 不符')
        return {'tcp_world': [round(float(v), 6) for v in p],
                'backoff_m': backoff, 'R_des_rowmajor': R.reshape(-1).tolist()}
    ck('target', _target)

    def _margin():
        _s = float(WGMPCConfig().joint_margin)
        s = _s
        E.check_margin_consistency(s, float(sc['endpoint']['joint_margin']))
        E.check_mode(sc['mode'])
        return {'solver': s, 'endpoint': float(sc['endpoint']['joint_margin']),
                'mode': sc['mode']}
    ck('margin_and_mode', _margin)

    def _dist():
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, 'gen_drawer_obstacle_params.py'),
             '--asset', ASSET, '--pose', f'{pose[0]},{pose[1]}',
             '--opening', '0.0', '--argv'],
            capture_output=True, text=True, check=True)
        args = r.stdout.strip().split('\n')
        if len(args) != 8:
            raise RuntimeError(f'產生器輸出 {len(args)} 個引數，應為 8')
        import ast as _a
        got = {}
        for i in range(0, len(args), 2):
            k, _, v = args[i + 1].partition(':=')
            got[k] = yaml.safe_load(v)
        if len(got['obstacles']) != 14:
            raise RuntimeError(f'具名幾何 {len(got["obstacles"])} 個，應為 14')
        if got['scene_truth'] is not True:
            raise RuntimeError('scene_truth 不是真')
        return {'n_obstacles': len(got['obstacles']),
                'n_approved': len(got['scene_truth_approved']),
                'pair_rows': got['pair_rows'], 'scene_truth': True}
    ck('distance_node_params', _dist)

    def _cfg_vs_runner():
        """設定檔與運行器的值必須一致 —— 兩邊漂移不會有任何錯誤訊息。"""
        import re
        src = open(os.path.join(HERE, 'run_wgmpc_stage_a.sh')).read()
        def lit(name):
            # **先試 ${NAME:-default} 形式**：先試裸值會把整個
            # `"${BACKOFF:-0.120}"` 當成值。
            # **不加 ^ 錨點**：同一行可能有兩個賦值
            #（PREPOS_VEL="…"; PREPOS_ACC="…"），加了就匹配不到第二個。
            for pat in (rf'(?:^|[;\s]){name}="\$\{{{name}:-([^}}]*)\}}"',
                        rf'(?:^|[;\s]){name}="([^"\n$]*)"',
                        rf'(?:^|[;\s]){name}=([^\s"$;]+)'):
                m = re.search(pat, src, re.M)
                if m:
                    return m.group(1)
            raise RuntimeError(f'運行器找不到 {name} 的字面值')
        pairs = [
            ('MODE', sc['mode'], lit('MODE')),
            # 起始路徑不一致 ⇒ 判準會被套錯一整組（交棒 vs 起始構型核對）
            ('START_MODE', sc['start_config']['start_mode'],
             lit('START_MODE')),
            ('JOINT_MARGIN', sc['endpoint']['joint_margin'],
             float(lit('JOINT_MARGIN'))),
            ('BACKOFF', sc['pre_contact_pause']['approach_backoff_m'],
             float(lit('BACKOFF'))),
            ('QUIET_S', sc['handover']['quiet_s'], float(lit('QUIET_S'))),
            ('SP_MEAS_TOL', sc['handover']['setpoint_meas_tol_rad'],
             float(lit('SP_MEAS_TOL'))),
            ('PREPOS_GOAL_TOL', sc['handover']['prepos_goal_tol_rad'],
             float(lit('PREPOS_GOAL_TOL'))),
            ('PREPOS_VEL', sc['prepos']['vel_max_rps'], float(lit('PREPOS_VEL'))),
            ('PREPOS_ACC', sc['prepos']['acc_max_rps2'], float(lit('PREPOS_ACC'))),
            ('W_S', sc['solver']['w_s'], float(lit('W_S'))),
            ('W_A', sc['solver']['w_a'], float(lit('W_A'))),
            ('W_S_ARM', sc['solver']['w_s_arm'], float(lit('W_S_ARM'))),
            ('NEAR_GAMMA', sc['solver']['near_target_gamma'],
             float(lit('NEAR_GAMMA'))),
            ('N', sc['solver']['horizon_N'], int(lit('N'))),
            ('RATE', sc['solver']['rate_hz'], float(lit('RATE'))),
            ('REACH_P', sc['pre_contact_pause']['tol_pos_m'],
             float(lit('REACH_P'))),
            ('REACH_R', sc['pre_contact_pause']['tol_rot_rad'],
             float(lit('REACH_R'))),
            ('HOLD_S', sc['pre_contact_pause']['hold_s'], float(lit('HOLD_S'))),
        ]
        bad2 = [f'{k}: 設定檔 {c!r} vs 腳本 {r!r}' for k, c, r in pairs
                if (abs(float(c) - float(r)) > 1e-12
                    if isinstance(c, (int, float)) and not isinstance(c, bool)
                    else str(c) != str(r))]
        if bad2:
            raise RuntimeError('設定檔與運行器不一致：' + '；'.join(bad2))
        return {'n_compared': len(pairs), 'all_match': True}
    ck('config_vs_runner', _cfg_vs_runner)

    def _handover():
        K = WholeBodyKinematics.from_urdf_file(
            os.path.join(WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
        lim = np.array(K.joint_limits())
        h = sc['handover']
        c = HandoverConfig(max_cmd_age_s=float(sc['endpoint']['max_cmd_age_s']),
                           quiet_s=float(h['quiet_s']),
                           max_state_age_s=float(h['max_state_age_s']),
                           joint_margin=float(sc['endpoint']['joint_margin']),
                           joint_lower=tuple(float(x) for x in lim[0, 3:]),
                           joint_upper=tuple(float(x) for x in lim[1, 3:]),
                           setpoint_meas_tol_rad=float(
                               h['setpoint_meas_tol_rad']))
        c.validate()
        return {'quiet_s': c.quiet_s, 'max_cmd_age_s': c.max_cmd_age_s,
                'u_prev': h['u_prev']}
    ck('handover_config', _handover)

    def _files():
        need = ['evaluation/run_wgmpc_stage_a.sh', 'evaluation/wgmpc_prepos_node.py',
                'evaluation/wgmpc_stage_a_handover.py', 'evaluation/wgmpc_wg2_node.py',
                'evaluation/arm_vel_adapter.py', 'evaluation/wgmpc_env_recorder.py',
                'evaluation/isaac_wholebody_sim_e2.py',
                'evaluation/results/wgmpc_arm_sp_ident_free4.json',
                'evaluation/models/omni_bot_wholebody_expanded.urdf',
                'evaluation/models/omni_bot_manip.urdf']
        miss = [f for f in need if not os.path.exists(os.path.join(WS, f))]
        if miss:
            raise RuntimeError(f'缺檔：{miss}')
        if not os.path.exists(os.path.join(WS, 'install/setup.bash')):
            raise RuntimeError('缺 install/setup.bash（需先 colcon build）')
        return {'all_present': True, 'n_checked': len(need) + 1}
    ck('files', _files)

    def _installed_matches_src():
        """**install/ 的套件必須與 src/ 一致。**

        `ros2 run ammr_wholebody_mpc …` 跑的是 install/ 裡的副本。先前只核了
        src/，於是場景真值來源（scene_truth_occ／validate_scene_truth）、
        安全層與交棒模組的改動**全都不在實際執行的節點裡** ——
        首趟因此在 [5/9] 以「讀不到 /arm_link_distance 的參數 scene_truth」中止。
        靜止基線沒起距離節點，所以一直沒暴露。

        依既定約定：ammr 的 python 套件**不得用 --symlink-install**
        （symlink 會弄壞 entry-point metadata，所有節點會以
        PackageNotFoundError 死掉）。所以只能靠重建，而重建就必須被核對。
        """
        import hashlib
        base_src = os.path.join(WS, 'src/ammr_wholebody_mpc/ammr_wholebody_mpc')
        base_ins = os.path.join(
            WS, 'install/ammr_wholebody_mpc/lib/python3.12/site-packages',
            'ammr_wholebody_mpc')
        if not os.path.isdir(base_ins):
            raise RuntimeError(f'找不到安裝目錄 {base_ins} ⇒ 需先 colcon build')
        names = [n for n in sorted(os.listdir(base_src)) if n.endswith('.py')]
        if not names:
            raise RuntimeError('src 下沒有 .py')
        diff, miss = [], []
        def sha(q):
            return hashlib.sha256(open(q, 'rb').read()).hexdigest()
        for n in names:
            a_, b_ = os.path.join(base_src, n), os.path.join(base_ins, n)
            if not os.path.exists(b_):
                miss.append(n); continue
            if sha(a_) != sha(b_):
                diff.append(n)
        if miss or diff:
            raise RuntimeError(
                f'install 與 src 不一致 ⇒ **需 colcon build**'
                f'（不得用 --symlink-install）；'
                f'缺 {miss[:5]}；內容不同 {diff[:5]}')
        # 本輪實際會用到的參數必須在**安裝版**裡宣告
        inst_dist = open(os.path.join(base_ins, 'arm_link_distance.py')).read()
        for key in ("p('scene_truth'", "p('scene_truth_approved'",
                    'def scene_truth_occ', 'def validate_scene_truth'):
            if key not in inst_dist:
                raise RuntimeError(f'安裝版的 arm_link_distance 缺 {key!r}')
        return {'n_modules': len(names), 'all_match': True,
                'scene_truth_declared_in_installed': True}
    ck('installed_matches_src', _installed_matches_src)

    def _scene_marker():
        src = open(os.path.join(HERE, 'run_wgmpc_stage_a.sh')).read()
        if 'articulation root' not in src:
            raise RuntimeError('運行器未使用已修正的場景就緒標記')
        if "grep -q '/joint_states\\|進入物理\\|physics'" in src:
            raise RuntimeError('運行器仍含會假命中的 physics 比對')
        return {'ready_marker': '[wb] articulation root',
                'false_match_removed': True}
    ck('scene_ready_marker', _scene_marker)

    rep['verdict'] = 'pass' if not bad else 'fail'
    rep['violations'] = bad
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1,
                  default=str)
    for k, v in rep.items():
        if k in ('verdict', 'violations'):
            continue
        print(f'  {k}: {v}')
    if bad:
        print('**入口核對未通過**：', file=sys.stderr)
        for b in bad:
            print(f'  - {b}', file=sys.stderr)
        return 1
    print('  入口核對全部通過')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
