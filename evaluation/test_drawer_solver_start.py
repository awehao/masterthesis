from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import wgmpc_wg2_node as solver


@pytest.mark.parametrize('first', ['release', 'align'])
def test_start_requires_both_signals_and_never_sends_a_warmup_command(monkeypatch, first):
    nd = SimpleNamespace(_drawer_released=False, _drawer_start=False,
                         _stop_req=None, drawer_ready_pub=Mock(), _publish_u=Mock())
    states = []

    def spin(**kwargs):
        states.append((nd._drawer_released, nd._drawer_start))
        if (len(states) == 1) == (first == 'release'):
            solver.WGMPCNode._on_drawer_handover(
                nd, SimpleNamespace(data='{"phase":"RELEASED"}'))
        else:
            solver.WGMPCNode._on_drawer_start(nd, SimpleNamespace(data=True))

    monkeypatch.setattr(solver.rclpy, 'ok', lambda: True)
    assert solver.wait_for_drawer_handover(nd, SimpleNamespace(spin_once=spin), 1) is None
    assert len(states) == 2
    assert states[0] == (False, False)
    nd._publish_u.assert_not_called()
    assert nd.drawer_ready_pub.publish.call_args.args[0].data is True


@pytest.mark.parametrize('reason', ['timeout', 'stop', 'shutdown'])
def test_wait_failure_never_starts_controller(monkeypatch, reason):
    nd = SimpleNamespace(_drawer_released=False, _drawer_start=False,
                         _stop_req='stop' if reason == 'stop' else None,
                         drawer_ready_pub=Mock(), _publish_u=Mock())
    monkeypatch.setattr(solver.rclpy, 'ok', lambda: reason != 'shutdown')
    why = solver.wait_for_drawer_handover(nd, Mock(), 0)
    assert why is not None
    nd._publish_u.assert_not_called()


@pytest.mark.parametrize('payload', ['{"phase":"HOLD"}', 'bad json', 'null'])
def test_nonrelease_or_invalid_message_revokes_release(payload):
    nd = SimpleNamespace(_drawer_released=True)
    solver.WGMPCNode._on_drawer_handover(nd, SimpleNamespace(data=payload))
    assert nd._drawer_released is False
