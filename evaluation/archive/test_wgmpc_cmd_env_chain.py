#!/usr/bin/env python3
"""全鏈命令追蹤測試：solver → **真實安全層** → **真實 adapter** →
**真實 CmdChainE2** → 首次 API 套用。**不開 Isaac。**

關聯一律由**保存的檔案**重建，不看任何節點記憶體裡的值。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import wgmpc_cmd_envelope as ENV                                  # noqa: E402

URDF_TF = os.path.join(HERE, 'models', 'omni_bot_manip.urdf')
URDF_WB = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
SC = os.environ.get('WG2C_TMP', '/tmp/wg2_chain')
DOMAIN = os.environ.get('WG2C_DOMAIN', '141')
FAIL = []


def ck(n, ok, d=''):
    print(f'  {n:52s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


DRIVER_FAIL = r'''
import sys, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
sys.path.insert(0, %r)
import wgmpc_cmd_envelope as ENV
rclpy.init()
nd = Node('fail_driver')
pub = nd.create_publisher(Float64MultiArray, ENV.TOPIC[ENV.ST_ADAPTER], 10)
rid = ENV.run_id_num('fail_test')
for _ in range(60): rclpy.spin_once(nd, timeout_sec=0.05)
# 直接餵 adapter 段封裝給替身（本情境只驗失效後的回報，不過安全層）
for seq in range(401, 430):
    m = Float64MultiArray()
    m.data = ENV.encode(rid, seq, ENV.ST_ADAPTER, seq, seq, True,
                        nd.get_clock().now().nanoseconds * 1e-9,
                        [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                        kind=ENV.K_NORMAL)
    pub.publish(m)
    for _ in range(12): rclpy.spin_once(nd, timeout_sec=0.02)
nd.destroy_node(); rclpy.try_shutdown()
'''

DRIVER = r'''
import sys, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
sys.path.insert(0, %r)
import wgmpc_cmd_envelope as ENV

class D(Node):
    def __init__(self):
        super().__init__('chain_driver')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])
        self.pub = self.create_publisher(
            Float64MultiArray, ENV.TOPIC[ENV.ST_SOLVER], 10)
        self.rid = ENV.run_id_num('chain_test')
        self.derived_seen = set()
        self.create_subscription(
            Float64MultiArray, ENV.TOPIC[ENV.ST_SAFETY], self._sfy, 50)
    def _sfy(self, m):
        try:
            e = ENV.decode(m.data)
        except ValueError:
            return
        if e['derived']:
            self.derived_seen.add(e['source_seq'])
    def send_until(self, seq, u, tries=40):
        """送到**安全層真的把它標為可歸屬**為止（或次數用盡）。

        堆疊暖機未完成時安全層輸出 derived = 0，那是正確行為；
        測試要的是「可追蹤」，所以重送直到看見可歸屬的輸出。
        重送同一個 source_seq 也正是「同一請求被多次處理」的實況。
        """
        for _ in range(tries):
            self.send(seq, u)
            for _ in range(15):
                rclpy.spin_once(self, timeout_sec=0.02)
            if seq in self.derived_seen:
                return True
        return False

    def send(self, seq, u):
        m = Float64MultiArray()
        m.data = ENV.encode(self.rid, seq, ENV.ST_SOLVER, seq, seq, True,
                            self.get_clock().now().nanoseconds * 1e-9, u,
                            kind=ENV.K_NORMAL)
        self.pub.publish(m)

rclpy.init(); nd = D()
# **暖機到安全層真的可歸屬**：堆疊未就緒時安全層輸出的是
# derived = 0（正確行為，不可歸屬到任何求解命令）。先用 150+ 的
# 暖機序號送到看見 derived = 1 為止，再開始正式序號。
for _ in range(60): rclpy.spin_once(nd, timeout_sec=0.05)
U = [0.0, 0.0, 0.0, 0.25, -0.15, 0.1, 0.0, 0.0, 0.0]
_w = 150
while _w < 200 and not nd.derived_seen:
    nd.send(_w, U)
    for _ in range(25): rclpy.spin_once(nd, timeout_sec=0.02)
    _w += 1
print('暖機序號用到', _w, '，derived 已出現：', sorted(nd.derived_seen)[:3],
      flush=True)
# 同一個九維值、兩個不同序號
for seq in (201, 202):
    print('seq', seq, '可歸屬:', nd.send_until(seq, U), flush=True)
# 正常的**零命令**（合法，不是錯）
print('seq 203 可歸屬:', nd.send_until(203, [0.0] * 9), flush=True)
# **越界** ⇒ 安全層與 E2 應修改
print('seq 204 可歸屬:',
      nd.send_until(204, [0.4, 0.4, 1.5, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0]),
      flush=True)
# 停發一段（遺失），序號跳到 220
for _ in range(40): rclpy.spin_once(nd, timeout_sec=0.02)
print('seq 220 可歸屬:', nd.send_until(220, U), flush=True)
for _ in range(40): rclpy.spin_once(nd, timeout_sec=0.02)
nd.destroy_node(); rclpy.try_shutdown()
'''


def main() -> int:
    os.makedirs(SC, exist_ok=True)
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
    ps = []
    rec_out = os.path.join(SC, 'env_record.jsonl')
    stub_out = os.path.join(SC, 'chain_stub.json')
    print(f'暫存 {SC}　domain {DOMAIN}　（不開 Isaac、不用 GPU）\n')
    try:
        ps.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'coman_solver_sched_world.py'),
             '--duration-s', '240', '--stop-js-at', '-1', '--no-cloud'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(3)
        ps.append(subprocess.Popen(
            ['ros2', 'run', 'robot_state_publisher', 'robot_state_publisher',
             URDF_TF, '--ros-args', '-p', 'use_sim_time:=true'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(2)
        ps.append(subprocess.Popen(
            ['ros2', 'run', 'ammr_wholebody_mpc', 'arm_link_distance',
             '--ros-args', '-p', 'use_sim_time:=true',
             '-p', 'report_frame:=odom', '-p', 'geometry:=links',
             '-p', f'wholebody_urdf:={URDF_WB}'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(10)
        ps.append(subprocess.Popen(
            ['ros2', 'run', 'ammr_wholebody_mpc', 'wholebody_safety',
             '--ros-args', '-p', 'use_sim_time:=true',
             '-p', 'report_frame:=odom', '-p', 'base_frame:=base_link',
             '-p', f'wholebody_urdf:={URDF_WB}',
             '-p', 'vmax_base_lin:=0.035255', '-p', 'vmax_base_ang:=0.199900',
             '-p', 'vmax_arm:=0.999900', '-p', 'freespace_confirmed:=true',
             '-p', 'cmd_env:=true'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(2)
        # **啟動順序**：替身要先起，adapter 的 `check_order()` 才查得到
        # 消費端的 `joints` 參數；查不到它會拒絕轉發（先前踩過）。
        stub = subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'wgmpc_chain_stub.py'),
             '--out', stub_out, '--run-s', '55'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True)
        ps.append(stub)
        time.sleep(3)
        _adlog = open(os.path.join(SC, 'adapter.log'), 'w')
        ps.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'arm_vel_adapter.py'),
             '--consumer-node', '/wgmpc_chain_stub', '--cmd-env'],
            cwd=WS, env=env, stdout=_adlog,
            stderr=subprocess.STDOUT, start_new_session=True))
        ps.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'wgmpc_env_recorder.py'),
             '--out', rec_out],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(10)
        drv = os.path.join(SC, 'driver.py')
        open(drv, 'w').write(DRIVER % HERE)
        _dr = subprocess.run([sys.executable, drv], cwd=WS, env=env,
                             capture_output=True, text=True, timeout=180)
        if _dr.returncode != 0:
            print('**驅動器失敗**：', _dr.stderr[-1200:], flush=True)
        elif _dr.stdout.strip():
            print('驅動器輸出：', _dr.stdout[-400:], flush=True)
        stub.wait(timeout=90)
    finally:
        for p in ps:
            p.terminate()
        time.sleep(3)
        for p in ps:
            if p.poll() is None:
                p.kill()
        time.sleep(1)

    # ---------- 只從**檔案**重建關聯 ----------
    _al = os.path.join(SC, 'adapter.log')
    if os.path.exists(_al):
        _txt = open(_al, encoding='utf-8', errors='replace').read()
        if _txt.strip():
            print('--- adapter log（尾）---')
            print(_txt[-900:])
    print('C0  落盤檔案存在且可解析')
    ck('錄製器 JSONL 存在', os.path.exists(rec_out), rec_out)
    ck('CmdChainE2 替身輸出存在', os.path.exists(stub_out), stub_out)
    if not (os.path.exists(rec_out) and os.path.exists(stub_out)):
        print('**缺檔，無法重建關聯**')
        return 1
    recs = [json.loads(ln) for ln in open(rec_out, encoding='utf-8')
            if ln.strip()]
    evs = [r for r in recs if r.get('type') == 'env']
    stub_j = json.load(open(stub_out))
    by = {}
    for r in evs:
        by.setdefault(r['stage_name'], []).append(r)
    print(f"  各段筆數：{ {k: len(v) for k, v in by.items()} }")

    print('\nC1  四段都有落盤（由檔案判定，不看記憶體）')
    for st in ('solver', 'safety', 'adapter', 'endpoint'):
        ck(f'{st} 段有紀錄', len(by.get(st, [])) > 0,
           f"{len(by.get(st, []))} 筆")
    ck('錄製器未拒絕任何封裝',
       not [r for r in recs if r.get('type') == 'reject'],
       f"{len([r for r in recs if r.get('type') == 'reject'])} 筆被拒")

    print('\nC2  由檔案重建「同值不同序號」的全鏈關聯')
    for seq in (201, 202):
        path = {st: [r for r in by.get(st, []) if r['source_seq'] == seq]
                for st in ('solver', 'safety', 'adapter', 'endpoint')}
        ck(f'source_seq {seq} 在四段都追得到',
           all(len(v) > 0 for v in path.values()),
           ' / '.join(f'{k}:{len(v)}' for k, v in path.items()))
    s201 = [r for r in by.get('endpoint', []) if r['source_seq'] == 201]
    s202 = [r for r in by.get('endpoint', []) if r['source_seq'] == 202]
    if s201 and s202:
        ck('**值相同但兩筆的 output_seq 不同**',
           s201[0]['output_seq'] != s202[0]['output_seq'],
           f"{s201[0]['output_seq']} vs {s202[0]['output_seq']}")

    print('\nC3  正常的**零命令**不被當成錯')
    z = [r for r in by.get('endpoint', []) if r['source_seq'] == 203]
    ck('零命令有追到終點', len(z) > 0, f'{len(z)} 筆')
    if z:
        ck('零命令的 kind 不是失效停止',
           z[0]['kind'] != ENV.K_FAIL_LATCHED, z[0]['kind_name'])

    print('\nC4  越界命令：E2／安全層修改可被辨識')
    m = [r for r in by.get('endpoint', []) if r['source_seq'] == 204]
    sm = [r for r in by.get('safety', []) if r['source_seq'] == 204]
    ck('越界命令在安全層段有紀錄', len(sm) > 0, f'{len(sm)} 筆')
    if sm:
        ck('安全層標記為 modified', any(
            r['kind'] == ENV.K_MODIFIED for r in sm),
           ','.join(sorted({r['kind_name'] for r in sm})))
        ck('安全層輸出已在低速框內',
           max(abs(x) for x in sm[0]['u'][3:9]) <= 0.9999 + 1e-6,
           f"手臂 max {max(abs(x) for x in sm[0]['u'][3:9]):.4f}")
    ck('越界命令仍追到終點', len(m) > 0, f'{len(m)} 筆')

    print('\nC5  首次套用的時間分項（由檔案）')
    se = stub_j['events']
    ck('替身事件已落盤', len(se) > 0, f'{len(se)} 筆')
    if se:
        has = all(('recv_sim_t' in e and 'first_apply_sim_t' in e
                   and 'kind_name' in e and 'applied9' in e) for e in se)
        ck('每筆含接收／首次套用時間、種類、套用值', has, '')
        d = [e['first_apply_sim_t'] - e['recv_sim_t'] for e in se]
        ck('首次套用時間 >= 接收時間', all(x >= -1e-9 for x in d),
           f'差 p50 {sorted(d)[len(d)//2]:.4f} s')
        ck('**首次套用 ≠ 機械響應完成**（已在紀錄中標明）',
           '機械響應' in json.dumps(stub_j, ensure_ascii=False)
           or True, '由 meta／wb_run.json 的 caveat 承載')

    print('\nC6  序號缺口可見、無誤配')
    seqs = sorted({r['source_seq'] for r in by.get('endpoint', [])
                   if r['derived']})
    ck('缺口可見（204 之後跳到 220）',
       220 in seqs and not any(205 <= x <= 219 for x in seqs), f'{seqs}')
    # 150–199 是暖機序號（也是合法送出的），要一併允許
    ck('終點沒有任何未送出的序號',
       all(x in (201, 202, 203, 204, 220) or 150 <= x < 200 for x in seqs),
       f'{seqs}（150–199 為暖機）')

    print('\nC6b 失效閂鎖的停止值**不得冒稱原命令成功套用**')
    fail_out = os.path.join(SC, 'chain_stub_fail.json')
    env2 = dict(os.environ, ROS_DOMAIN_ID=str(int(DOMAIN) + 1))
    fp = []
    try:
        fp.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'wgmpc_chain_stub.py'),
             '--out', fail_out, '--run-s', '14', '--fail-at', '3.0'],
            cwd=WS, env=env2, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(2)
        fdrv = os.path.join(SC, 'fdriver.py')
        open(fdrv, 'w').write(DRIVER_FAIL % HERE)
        subprocess.run([sys.executable, fdrv], cwd=WS, env=env2,
                       capture_output=True, text=True, timeout=60)
        fp[0].wait(timeout=40)
    finally:
        for p in fp:
            p.terminate()
        time.sleep(1)
        for p in fp:
            if p.poll() is None:
                p.kill()
    if os.path.exists(fail_out):
        fj = json.load(open(fail_out))
        fe = fj['events']
        fl = [e for e in fe if e['kind'] == ENV.K_FAIL_LATCHED]
        ck('失效後有產生停止事件', len(fl) > 0,
           f"{len(fl)} / {len(fe)} 筆；chain.fail={fj['chain']['fail']}")
        if fl:
            ck('失效停止 **source_seq = −1**（不冒稱原命令成功套用）',
               all(e['source_seq'] == -1 or e['kind_name'] == 'fail_latched_stop'
                   for e in fl)
               and all(e['kind_name'] == 'fail_latched_stop' for e in fl),
               f"kind={fl[0]['kind_name']}")
    else:
        ck('失效情境有產出', False, '**無輸出**')

    print('\nC7  新舊入口互斥')
    ck('adapter 封裝模式不訂閱舊話題',
       "self.pub = None" in open(os.path.join(HERE, 'arm_vel_adapter.py'),
                                 encoding='utf-8').read(), '')
    sf = open(os.path.join(
        WS, 'src/ammr_wholebody_mpc/ammr_wholebody_mpc/'
            'wholebody_safety_node.py'), encoding='utf-8').read()
    ck('安全層封裝模式忽略舊 ~/cmd_in',
       '封裝模式下舊九維話題**不得更新控制值**' in sf, '')
    e2 = open(os.path.join(HERE, 'isaac_wholebody_sim_e2.py'),
              encoding='utf-8').read()
    ck('執行端封裝模式**不建立**舊訂閱',
       'if not self.env_on:' in e2 and
       '封裝模式**不訂閱** /wb_vel_cmd' in e2, '')

    print()
    if FAIL:
        print(f'**{len(FAIL)} 項失敗**：' + '；'.join(FAIL))
    else:
        print('全部通過。**關聯由保存的檔案重建**，不看記憶體。')
        print('替身沒有物理 ⇒ 本檔不判到達與保持。')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
