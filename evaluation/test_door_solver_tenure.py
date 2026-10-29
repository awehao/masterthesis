import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from door_solver_tenure import SolverTenure  # noqa: E402

U = [0.02, 0.0, 0.0] + [0.0] * 6


def ann(step, t, **kw):
    d = {'run_id': 'R', 'step': step, 'sim_t': t, 'owner': 'wholebody', 'chain_tenure': 0, 'chain_tenure_open': True,
         'expected_tenure': 0, 'pending_to': None, 'arm_hold': [0, 0, 0.15, 0, -1.1, 0], 'direct_applied': None}
    d.update(kw)
    return d


def closed_and_filled(T=1.0):
    s = SolverTenure('R')
    s.on_announce(ann(10, 0.1, owner='nav', chain_tenure_open=False, expected_tenure=1))
    assert s.state == 'closed' and not s.publish_allowed()
    for i in range(int(T / 0.01)):
        s.on_applied(11 + i, 0.11 + i * 0.01, U, True, 4)
    return s


def test_close_rearm_ok():
    s = closed_and_filled()
    last = s.hist[-1]
    s.on_announce(ann(last[0], last[1], owner='nav', chain_tenure_open=False, expected_tenure=1, pending_to='wholebody',
                      direct_applied={'u': list(U), 'sim_t': last[1], 'step': last[0]}))
    r, h = s.try_rearm(last[1] + 0.01, 0.13)
    assert r == 'ok' and s.k == 1 and s.publish_allowed() and len(h) >= 13


def test_wait_then_fail():
    s = closed_and_filled(0.05)
    last = s.hist[-1]
    s.on_announce(ann(last[0], last[1], owner='nav', chain_tenure_open=False, expected_tenure=1, pending_to='wholebody',
                      direct_applied={'u': list(U), 'sim_t': last[1], 'step': last[0]}))
    r, why = s.try_rearm(last[1] + 0.01, 0.13)
    assert r == 'wait' and '不足' in why
    r, why = s.try_rearm(last[1] + 3.0, 0.13)
    assert r == 'fail' and s.state == 'failed' and not s.publish_allowed()


def test_mismatch_and_wrong_run():
    s = closed_and_filled()
    last = s.hist[-1]
    s.on_announce(ann(last[0], last[1], owner='nav', chain_tenure_open=False, expected_tenure=1, pending_to='wholebody',
                      direct_applied={'u': [0.03] + U[1:], 'sim_t': last[1], 'step': last[0]}))
    assert s.try_rearm(last[1] + 0.01, 0.13)[0] == 'wait'
    assert not s.on_announce(dict(ann(last[0] + 1, last[1]), run_id='X'))
    assert not s.on_announce(ann(last[0] - 5, last[1]))


def test_not_applied_reports_ignored():
    s = SolverTenure('R')
    s.on_applied(1, 0.0, U, True, 0)
    assert not s.hist
    s.on_announce(ann(2, 0.02, owner='nav', chain_tenure_open=False, expected_tenure=1))
    s.on_applied(3, 0.03, U, False, 4)
    assert not s.hist
