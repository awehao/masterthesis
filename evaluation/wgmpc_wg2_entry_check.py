#!/usr/bin/env python3
"""下一趟（WG2 增廣模型自由空間錄影驗證）的**入口核對**。

**不啟動 Isaac、不用 GPU。** 只做靜態與離線核對：
runner 真的傳了模型選擇與錄影參數、控制參數與 free4 一致、
取景涵蓋底盤／手臂／目標、封存判定是內容核對、保護未放寬。
"""
from __future__ import annotations

import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (            # noqa: E402
    WholeBodyKinematics)

RUNNER = os.path.join(HERE, 'run_wgmpc_wg2_free.sh')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
FREE4 = os.path.join(HERE, 'runs', 'wgmpc_wg2_free4_182234')
FAIL = []
REP = {}


def ck(n, ok, d=''):
    print(f'  {n:52s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    REP.setdefault('checks', []).append({'name': n, 'ok': bool(ok),
                                         'detail': str(d)})
    if not ok:
        FAIL.append(n)


def shv(src, name):
    """取 shell 變數的最後一次賦值。

    **只認賦值位置**（行首或 `;` 之後），不認任意空白之後 ——
    否則 run-label 裡的 `N=5（...）` 這種字串也會被當成賦值
    （實測踩過：N 被讀成「5（$ARM_MODEL 模型）到達與保持」）。
    """
    m = re.findall(rf'(?:^|;\s*){re.escape(name)}="?([^"\n;]*)"?', src, re.M)
    if not m:
        return None
    v = m[-1]
    # 支援 `VAR="${VAR:-預設值}"` 的寫法：取預設值。
    # （先前只認字面賦值，遇到這種形式會把 '${REC_TARGET:-0.44700' 當成值。）
    d = re.fullmatch(r'\$\{' + re.escape(name) + r':-(.*)\}', v)
    return d.group(1) if d else v


def eff(src, name):
    """**實際會生效的值**：環境變數優先，其次是 runner 裡的字面／預設值。

    遠目標趟次由環境變數傳 ABS_TARGET / REC_TARGET / REC_AT / REC_EYE，
    只讀 runner 原始碼會核到近目標的設定，與真正要跑的那一組不同。
    """
    v = os.environ.get(name)
    return v if v not in (None, '') else shv(src, name)


def code_only(src):
    """去掉註解與 `say`／`echo` 的訊息字串，只留可執行碼。

    危險樣式（pkill、自動重跑）只能掃可執行碼：runner 的註解裡寫著
    「不用 pkill」、訊息裡寫著「不自動重跑」，拿原文掃會把**說明**
    誤判成**行為**（實測踩過）。
    """
    out = []
    for ln in src.split('\n'):
        t = re.sub(r'(?<!\\)#.*$', '', ln)
        t = re.sub(r'\b(say|echo)\s+"[^"]*"', r'\1 ""', t)
        out.append(t)
    return '\n'.join(out)


def main() -> int:
    src = open(RUNNER, encoding='utf-8').read()
    print('=== 入口核對：WG2 增廣模型自由空間錄影驗證 ===')
    print('**本核對不啟動 Isaac、不用 GPU。**\n')

    print('A  模型選擇確實傳入')
    ck('runner 傳 --arm-model', '--arm-model "$ARM_MODEL"' in src,
       f'ARM_MODEL={shv(src, "ARM_MODEL")}')
    # 原本比對整串 `${ARM_MODEL:-setpoint}` —— 那是因為舊的 shv 無法解析
    # `${VAR:-預設}`，把整串當成值。shv 修好後要分兩件事核：
    #   (a) 寫法仍是**可由環境覆寫的預設**（保留這個契約）
    #   (b) 解析出來的預設值是 setpoint，不是節點預設的 ideal
    ck('預設值為 setpoint（不是節點預設的 ideal）',
       'ARM_MODEL="${ARM_MODEL:-setpoint}"' in src
       and shv(src, 'ARM_MODEL') == 'setpoint',
       str(shv(src, 'ARM_MODEL')))
    ck('runner 傳 --arm-ident', '--arm-ident "$ARM_IDENT"' in src, '')
    ck('辨識檔存在且可解析', os.path.exists(IDENT), IDENT)
    if os.path.exists(IDENT):
        idn = json.load(open(IDENT))
        al, bs = np.array(idn['alpha']), np.array(idn['bias_rad'])
        ck('辨識參數有限且 α ∈ (0,1)',
           bool(np.isfinite(al).all() and np.isfinite(bs).all()
                and (al > 0).all() and (al < 1).all()),
           f'α p50 {np.median(al):.5f}　|b| max {np.abs(bs).max():.3e}　'
           f'phys_dt {idn["phys_dt_measured_s"]}')
        REP['ident'] = {'alpha_p50': float(np.median(al)),
                        'phys_dt_s': idn['phys_dt_measured_s'],
                        'status': idn.get('status')}
    ck('啟動配置會落盤（含實際模型選擇）',
       "'arm_model': '$ARM_MODEL'" in src and 'run_config.json' in src, '')
    ck('缺辨識檔時拒絕啟動', 'exit 66' in src, '')
    ck('節點預設仍是 ideal（既有配置不受影響）',
       "'--arm-model', default='ideal'" in open(
           os.path.join(HERE, 'wgmpc_wg2_node.py'), encoding='utf-8').read(),
       '')

    print('\nB  控制參數與 free4 一致（不調 N、權重、目標、限制）')
    w4 = json.load(open(os.path.join(FREE4, 'wg2_out.json')))
    a4 = w4['args']
    for nm, got, exp in (('N', shv(src, 'N'), str(a4['N'])),
                         ('RATE', shv(src, 'RATE'), str(int(a4['rate']))),
                         ('REACH_P', shv(src, 'REACH_P'),
                          str(a4['reach_pos_m'])),
                         ('REACH_R', shv(src, 'REACH_R'),
                          str(a4['reach_rot_rad'])),
                         ('HOLD_S', shv(src, 'HOLD_S'), str(a4['hold_s'])),
                         ('SIM_LIMIT', shv(src, 'SIM_LIMIT'), '120')):
        ck(f'{nm} 與 free4 相同', got == exp, f'{got} vs free4 {exp}')
    ck('OFFSET 與 free4 相同',
       [float(x) for x in (shv(src, 'OFFSET') or '').split()]
       == list(a4['target_offset']),
       f'{shv(src, "OFFSET")} vs {a4["target_offset"]}')
    for nm, exp in (('VMAX_BASE_LIN', '0.035255'),
                    ('VMAX_BASE_ANG', '0.199900'),
                    ('VMAX_ARM', '0.999900')):
        ck(f'L1 低速框 {nm}', shv(src, nm) == exp,
           f'{shv(src, nm)}（核准值 {exp}）')
    ck('u_prev 政策仍為 strict', '--u-prev-policy strict' in src, '')
    ck('freespace_confirmed 仍為 true',
       'freespace_confirmed:=true' in src, '')

    print('\nC  時間預算：錄影會拖慢 sim:wall，故以模擬時間與 free4 對齊')
    s4 = json.load(open(os.path.join(FREE4, 'sim', 'wb_run.json')))
    pub4 = [x for x in w4['log'] if x.get('published')]
    span4 = pub4[-1]['sim_t'] - pub4[0]['sim_t']
    ck('節點收到 --duration-sim-s', '--duration-sim-s "$TASK_SIM_S"' in src,
       f'TASK_SIM_S={shv(src, "TASK_SIM_S")}')
    ck('模擬時間預算 ≈ free4 的任務跨度',
       abs(float(shv(src, 'TASK_SIM_S')) - span4) <= 1.0,
       f'{shv(src, "TASK_SIM_S")} s vs free4 {span4:.2f} s')
    ck('牆鐘上限留寬（容許 render 拖慢）',
       float(shv(src, 'TASK_WALL_S')) >= 3 * float(shv(src, 'TASK_SIM_S')),
       f'牆鐘 {shv(src, "TASK_WALL_S")} s；free4 sim:wall '
       f'= {s4["sim_time_s"]/s4["wall_s"]:.3f}（無錄影）')
    REP['free4_task_sim_span_s'] = round(span4, 2)

    print('\nD  錄影：模擬器內相機、1280×720、10 fps')
    ck('解析度為 1280x720（非預設 1920x1080）',
       shv(src, 'REC_RES') == '1280x720', str(shv(src, 'REC_RES')))
    ck('fps 為 10（非預設 30）', shv(src, 'REC_FPS') == '10',
       str(shv(src, 'REC_FPS')))
    e2 = open(os.path.join(HERE, 'isaac_wholebody_sim_e2.py'),
              encoding='utf-8').read()
    ck('錄影來源是**模擬器內相機**（prim /World/rec_cam）',
       "Camera(prim_path='/World/rec_cam'" in e2
       and 'isaacsim.sensors.camera' in e2, '')
    ck('**無桌面錄製**（無 screen/grim/ffmpeg x11 等）',
       not re.search(r'grim|wf-recorder|x11grab|import -window|scrot', e2), '')
    ck('每 N 步取樣由 fps 與 physics_dt 算出',
       'rec_every = max(1, int(round(1.0 / (a.record_fps * a.physics_dt))))'
       in e2, '10 fps × 0.01 s ⇒ 每 10 個物理步取 1 格')
    ck('目標標記為純視覺（無 CollisionAPI）⇒ 不觸發 freespace 中止',
       'UsdGeom.Sphere.Define(stage, \'/World/rec_target/point\')' in e2
       and 'CollisionAPI' not in e2.split('rec_target')[1][:1800], '')

    print('\nE  取景涵蓋底盤、手臂與目標（八角點投影，不靠外接球）')
    K = WholeBodyKinematics.from_urdf_file(
        os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    links = sorted({j.child for j in K.joints.values()} | {'base_link'})
    P = np.array([K.fk(np.zeros(9), L)[:3, 3] for L in links])
    ci = {c: i for i, c in enumerate(s4['log_cols'])}
    L4 = np.array(s4['log'], float)
    tcp = L4[:, [ci['tcp_x'], ci['tcp_y'], ci['tcp_z']]]
    bas = L4[:, [ci['base_x'], ci['base_y']]]
    T0 = K.fk(np.zeros(9), 'link_tcp')[:3, 3]
    _abs = eff(src, 'ABS_TARGET')
    if _abs:
        # **遠目標**：目標與包絡都不能沿用近目標的。包絡取自已登錄的
        # B 規格（離線預覽軌跡逐連桿 FK），不在此重跑閉迴路。
        tgt = np.array([float(x) for x in _abs.split()])
        bspec = os.path.join(HERE, 'results', 'wgmpc_wg2_far_target_B_spec.yaml')
        if not os.path.exists(bspec):
            FAIL.append('遠目標趟次但找不到 B 規格：' + bspec)
            lo, hi = tgt - 0.5, tgt + 0.5
        else:
            import yaml
            fr = yaml.safe_load(open(bspec))['framing']
            lo = np.array(fr['envelope_lo'], float)
            hi = np.array(fr['envelope_hi'], float)
            ck('遠目標與 B 規格一致',
               np.linalg.norm(tgt - np.array(
                   yaml.safe_load(open(bspec))['target']['world_m'])) < 1e-9,
               f'ABS_TARGET={tgt.tolist()}')
    else:
        tgt = T0 + np.array([float(x) for x in shv(src, 'OFFSET').split()])
        PAD = 0.12
        pts = np.vstack([P, tcp, np.column_stack([bas, np.zeros(len(bas))]),
                         tgt[None, :]])
        lo, hi = pts.min(0) - PAD, pts.max(0) + PAD
    lo[2] = max(0.0, lo[2])
    at = np.array([float(x) for x in eff(src, 'REC_AT').split(',')])
    eye = np.array([float(x) for x in eff(src, 'REC_EYE').split(',')])
    corners = np.array(list(itertools.product(*zip(lo, hi))))

    def frac(vf_deg, margin=0.12):
        hf = 2 * math.degrees(math.atan(
            math.tan(math.radians(vf_deg / 2)) * 16 / 9))
        f = at - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0., 0., 1.])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        th = math.tan(math.radians(hf / 2)) * (1 - margin)
        tv = math.tan(math.radians(vf_deg / 2)) * (1 - margin)
        wh = wv = 0.0
        for c in corners:
            v = c - eye
            z = float(v @ f)
            if z <= 1e-6:
                return 9.0, 9.0
            wh = max(wh, abs(float(v @ r)) / z / th)
            wv = max(wv, abs(float(v @ u)) / z / tv)
        return wh, wv

    d = float(np.linalg.norm(eye - at))
    ck('包絡含 URDF 連桿、' + ('B 規格離線軌跡' if _abs else 'free4 軌跡')
       + '、底盤與目標', True,
       f'lo {np.round(lo,3)} hi {np.round(hi,3)}　相機距離 {d:.2f} m'
       + ('　**遠目標**' if _abs else ''))
    # **FOV 已由 rec5 實拍反推**（≈26–27.6°，與光圈假設相符）；
    # 判定改用 24° 起跳的實測範圍，不再用無依據的悲觀 20°。
    for vf in (24.0, 26.0, 27.6):
        wh, wv = frac(vf)
        ck(f'vFOV {vf:.1f}° 下八角點全在框內',
           wh <= 1.0 and wv <= 1.0,
           f'水平佔 {wh*100:.0f}%、垂直佔 {wv*100:.0f}% 的可用框')
    ck('目標標記與名目目標一致（權威值由節點趟中寫入）',
       np.linalg.norm(
           np.array([float(x) for x in eff(src, 'REC_TARGET').split(',')])
           - tgt) < 0.001,
       f'標記 {eff(src, "REC_TARGET")}　名目 {np.round(tgt,5)}'
       + ('' if _abs else f'；free4 實錄差 '
          f'{np.linalg.norm(tgt - np.array(w4["target_tcp"])):.4f} m '
          f'< 標記半徑 0.020 m'))
    REP['framing'] = {'at': at.tolist(), 'eye': eye.tolist(),
                      'distance_m': round(d, 3),
                      'envelope_lo': lo.tolist(), 'envelope_hi': hi.tolist(),
                      'fov_basis':
                      'rec5 趟次實拍影格反推 vFOV ≈ 26–27.6°，'
                      '與「水平光圈 20.955 mm、焦距 24 mm、16:9」一致；'
                      '距離以偏保守的 24° 訂（而非無依據的 20°）'}

    print('\nF  趟次目錄與封存判定')
    ck('free4 仍完整保留', os.path.exists(
        os.path.join(FREE4, 'sim', 'wb_run.json')), FREE4)
    ck('RUN_ID 每趟獨立（帶時間戳）',
       'RUN_ID="${RUN_ID:-wgmpc_wg2_free_$(date +%H%M%S)}"' in src, '')
    ck('封存判定**不是**「有任意 JSON」',
       'ls "$DIR/sim/"*.json' not in src
       and 'wgmpc_wg2_archive_check.py' in src, '')
    ck('封存核對要求錄影與模型選擇',
       '--expect-recording' in src and '--expect-arm-model "$ARM_MODEL"'
       in src, '')
    for tag, args, want in (
            ('legacy 趟次不誤判為缺陷', [FREE4], 0),
            ('legacy ＋ 要求新模型 ⇒ 擋下', [FREE4, '--expect-arm-model',
                                        'setpoint'], None),
            ('要求錄影但沒有 ⇒ 72', [FREE4, '--expect-recording'], 72)):
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, 'wgmpc_wg2_archive_check.py')]
            + args, cwd=WS, capture_output=True, text=True)
        ck(tag, (r.returncode == want) if want is not None
           else (r.returncode != 0), f'exit {r.returncode}')

    print('\nG  保護未放寬')
    ck('熱中止線仍為 92（執行端）', 'CPU_LIMIT_C = 92.0' in e2
       or re.search(r'cpu[-_]limit[^\n]*92', e2) is not None,
       re.findall(r'[\w.\-]*92\.0[\w]*', e2)[:2])
    code = code_only(src)
    ck('cleanup 只針對本趟記錄的 PID',
       'for i in "${!PIDS[@]}"' in code
       and 'kill -TERM "${PIDS[$i]}"' in code, '只對 PIDS[] 逐一送信號')
    ck('可執行碼無 pkill／killall／廣泛比對殺程序',
       not re.search(r'pkill|killall|kill\s+-\w+\s+\$\(p[sg]', code),
       '（註解裡的「不用 pkill」不計入）')
    ck('要求 ROS_DOMAIN_ID（獨立 domain）',
       'ROS_DOMAIN_ID 未設定，拒絕啟動' in src, '')
    ck('起動前檢查仍在（clock／讀回／通路核對）',
       'clock_advancing.py' in src and 'coman_lowspeed_readback.py' in src
       and 'wgmpc_wg2_freespace_check.py' in src, '')
    ck('可執行碼不含自動重跑迴圈',
       not re.search(r'(for|while)\b[^\n]*\b(retry|RETRY|re_?run)\b', code)
       and not re.search(r'bash\s+evaluation/run_wgmpc', code),
       '（訊息裡的「不自動重跑」不計入）')
    ck('保護觸發後不自動重跑（封存不完整僅標記並保留）',
       '不自動重跑' in src and 'archive_complete": false' in src, '')

    print('\nH  資源')
    du = shutil.disk_usage(HERE)
    n_max = int(120 * float(shv(src, 'REC_FPS')))
    ck('磁碟餘裕足夠（估 1280×720 PNG ≤ 1.0 MB／格）',
       du.free / 1e9 > 5.0,
       f'可用 {du.free/1e9:.1f} GB；最多 {n_max} 格 ⇒ 估 ≤ {n_max*1.0/1e3:.1f} GB')
    ck('bash -n 通過', subprocess.run(['bash', '-n', RUNNER]).returncode == 0,
       '')

    print()
    if FAIL:
        print(f'**{len(FAIL)} 項失敗**：' + '；'.join(FAIL))
    else:
        print('入口核對全部通過。**Isaac 未啟動。**')
        print('等使用者重新確認插電、人在機旁後，只跑一趟；'
              '不自動重跑、不放寬門檻。')
    REP['pass'] = not FAIL
    out = os.path.join(HERE, 'results', 'wgmpc_wg2_entry_check.json')
    json.dump(REP, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f'-> {out}')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
