import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_w2_tenure_auth as TA  # noqa: E402


class Stub:
    """模擬 ModeAuthority3／4 的兩個提交後逾時。"""
    def __init__(self):
        self.post_commit_timeout_s, self.applied_timeout_s = 0.25, 0.25
        self.commit = {'step': 0, 'applied_confirmed': True}
        self.pending = self.cand = None
        self.aborted = None
        self.mode_seq = 4
        self.last_applied_t = 0.0
        self.last_applied_step = 0
        self.cur_stamp = 0.0
        self.rejections = []

    def abort(self, why, step, t):
        self.aborted = why

    def _nack(self, why, msg, step, t):
        self.rejections.append(why)
        return 'reject', why

    def select(self, step, t, u):
        if t - self.last_applied_t > self.applied_timeout_s:
            self.abort('post_commit_not_applied', step, t)
            return None
        if t - self.cur_stamp > self.post_commit_timeout_s:
            self.abort('post_commit_source_lost', step, t)
            return None
        return 'cur'

    def on_request(self, msg, step, t):
        return 'pending', None

    def stats(self):
        return {}


def test_without_suspend_aborts():
    a = TA.tenure_suspend(Stub)()
    a.select(1, 0.3, None)
    assert a.aborted == 'post_commit_not_applied'


def test_suspend_resume():
    a = TA.tenure_suspend(Stub)()
    assert a.tenure_suspend(10, 0.1) == (True, None)
    for i in range(500):
        assert a.select(11 + i, 0.11 + i * 0.01, None) == 'cur'
    assert a.aborted is None and a.post_commit_timeout_s == 0.25 and a.applied_timeout_s == 0.25
    assert a.on_request({'req_seq': 5}, 600, 5.2)[0] == 'reject' and a.rejections == ['tenure_closed']
    a.cur_stamp = 5.3
    assert a.tenure_resume(600, 5.3, 1) == (True, None)
    assert a.last_applied_t == 5.3 and a.select(601, 5.31, None) == 'cur'
    a.select(700, 5.7, None)
    assert a.aborted == 'post_commit_not_applied'            # 恢復後原逾時照常


def test_guards():
    a = TA.tenure_suspend(Stub)()
    a.commit = {'step': 0, 'applied_confirmed': None}
    assert a.tenure_suspend(1, 0.1)[0] is False and a.aborted == 'tenure_suspend_without_confirmed_commit'
    b = TA.tenure_suspend(Stub)()
    b.tenure_suspend(1, 0.1)
    assert b.tenure_resume(2, 0.2, 2)[0] is False and b.aborted.startswith('tenure_resume_mismatch')


def test_real_class_has_attributes():
    import door_regrasp_mode3 as RGM3
    C = TA.tenure_suspend(RGM3.RegraspAuthority)
    a = C('R', run_id_num=1.0, stage_id=2, cand_timeout_s=1.0, cmd_max_age_s=0.2, post_commit_timeout_s=0.25, dt=0.01,
          a_lin=0.5, a_ang=2.0, arm_rate_max=1.0)
    for k in ('post_commit_timeout_s', 'applied_timeout_s', 'commit', 'pending', 'cand', 'mode_seq', 'aborted', 'last_applied_t'):
        assert hasattr(a, k), k
    assert 'tenure_auth' in a.stats()
