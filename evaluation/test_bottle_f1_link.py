#!/usr/bin/env python3
"""候選連結輪次切換的單元核對（SegTargetLink.rebind；不用 ROS、Isaac）：切換前的列與完成通知到不了新接收者；切換後新序號（從 1 重來）照常轉送。
1 切換後才到的舊列（t_eval 不大於已處理的最新值）⇒ t_eval_not_increasing，不轉送。
2 切換後才到的舊完成通知（新實例沒有對應的最新列）⇒ 不補送。
3 切換後的新列序號 1 ⇒ 轉送給新接收者，舊接收者收不到。
    python3 evaluation/test_bottle_f1_link.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_segtarget_link as SL                      # noqa: E402

CF = ('t_eval', 'src_stamp', 'k_ready', 'k_age_s', 'k_snap_seq', 'k_snap_t_last', 'k_snap_cx', 'k_snap_cy', 'k_snap_z_support', 'k_snap_ik', 'k_snap_local', 'k_snap_dev_mm', 'k_snap_consistent')
LB = 'bottle_target_candidates schema=3 frame=odom stamps=ros fields=' + ','.join(CF)
KF = ('t_publish', 'kind', 'seq', 't_first', 't_last', 'n', 'cx', 'cy', 'z_support', 'z_grasp', 'ik', 'local', 'local_min', 'schema')
LK = 'bottle_target_candidate_checks schema=3 frame=odom kind=0 complete,1 known_type stamps=ros fields=' + ','.join(KF)


class Node:
    def create_subscription(self, *a, **k):
        return None


def row(te, seq, t_last, ik=2.0, x=0.15):
    return (LB, [te, te - 0.01, 1.0, 0.05, float(seq), t_last, x, 0.42, 0.60, ik, ik, 0.0, 1.0])


def chk(tp, seq, x=0.15):
    return (LK, [tp, 1.0, float(seq), tp - 0.3, tp - 0.1, 3.0, x, 0.42, 0.60, 0.70, 2.0, 2.0, 0.06, 3.0])


def main():
    old, new = [], []
    L = SL.SegTargetLink(Node(), None, None, sink=lambda t, c: old.append((t, c)))
    L.set_T0(0.0)
    L.q.append(row(10.0, 3, 9.9, ik=1.0))                 # 第一輪：序號 3 檢查中
    L.tick(10.1)
    n_old = len(old)
    L.rebind(None, lambda t, c: new.append((t, c)))
    L.q.append(row(9.5, 2, 9.4))                          # 1 舊列晚到
    L.qc.append(chk(10.05, 3))                            # 2 舊完成通知晚到
    L.tick(10.2)
    c = {}
    c['舊列不轉送'] = not new and L.rej.get('t_eval_not_increasing', 0) >= 1
    c['舊通知不補送'] = not new
    L.q.append(row(12.0, 1, 11.9, x=-0.15))               # 3 新實例序號從 1
    L.tick(12.1)
    c['新列轉送給新接收者'] = len(new) == 1 and new[0][1]['snap_seq'] == 1.0 and new[0][1]['snap_cx'] == -0.15
    c['舊接收者不再收到'] = len(old) == n_old
    for k, v in c.items():
        print(('✓' if v else '✗'), k)
    print('全過' if all(c.values()) else f'未過；拒收 {L.rej}')
    return 0 if all(c.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
