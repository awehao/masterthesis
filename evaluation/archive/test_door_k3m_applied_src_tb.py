import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from door_k3m_applied_src_tb import EnvIntakeTB  # noqa: E402


class Chain:
    def __init__(self):
        self.n_recv, self.got = 0, []

    def receive(self, u, t):
        self.n_recv += 1
        self.got.append(list(u))


def env(seq, vx, tenure=1):
    return {'stage_id': 2, 'kind': 0, 'source_seq': seq, 'derived': True, 'u': [vx, 0.0, 0.0] + [0.0] * 6, 'tenure': tenure, 'mode_seq': 4}


def mk():
    it = EnvIntakeTB(run_id_num=None, stage_id=2, kinds=(0,))
    it.set_expected(1, 0.0)
    return it


def test_micro_excess_snapped_and_repeat_is_duplicate():
    it, ch = mk(), Chain()
    e = env(5, 0.035255 + 3e-9)
    assert it.receive(e, ch, 1.0) == 'fed' and ch.got[-1][0] == 0.035255
    assert e['u'][0] == 0.035255 + 3e-9                       # 權威持有的原封裝不改
    assert it.receive(e, ch, 1.01) == 'duplicate'              # 同一筆再餵：修正確定性 ⇒ 冪等
    assert it.n_snapped == 2 and it.report()['box_snap']['log'][0]['source_seq'] == 5


def test_large_excess_not_snapped():
    it, ch = mk(), Chain()
    assert it.receive(env(6, 0.0360), ch, 1.0) == 'fed'
    assert ch.got[-1][0] == 0.0360 and it.n_over == 1 and it.n_snapped == 0


def test_inside_box_untouched_and_negative_side():
    it, ch = mk(), Chain()
    it.receive(env(7, 0.0300), ch, 1.0)
    assert ch.got[-1][0] == 0.0300
    it.receive(env(8, -(0.035255 + 5e-7)), ch, 1.1)
    assert ch.got[-1][0] == -0.035255
