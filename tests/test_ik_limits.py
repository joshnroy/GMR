"""GeneralMotionRetargeting must pass its IK limits to mink.solve_ik.

mink's signature is ``solve_ik(configuration, tasks, dt, solver, damping=1e-12, safety_break=False, limits=None, ...)``.
Passing ``self.ik_limits`` as the sixth positional argument put the limits into ``safety_break``; mink then fell back to
its default (joint limits only), so any extra limit, such as the velocity limit, was silently dropped.
"""

import mink
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from skeleton import STANDING


def _frame(seed):
    """The standing skeleton with random body orientations (wxyz), a pose far from the robot's start."""
    rot = R.random(random_state=seed).as_quat(scalar_first=True)
    return {k: [np.array(p), rot.copy()] for k, p in STANDING.items()}


def _spy(monkeypatch):
    calls = []
    solve_ik = mink.solve_ik

    def spy(*args, **kwargs):
        vel = solve_ik(*args, **kwargs)
        calls.append((args, kwargs, vel))
        return vel

    monkeypatch.setattr(mink, "solve_ik", spy)
    return calls


def test_limits_passed_by_keyword(monkeypatch):
    calls = _spy(monkeypatch)
    retargeter = GMR("bvh_lafan1", "sprout", verbose=False)
    retargeter.retarget(_frame(1))
    assert calls
    for args, kwargs, _ in calls:
        assert len(args) <= 5, "nothing may land in solve_ik's safety_break slot"
        assert kwargs.get("limits") is retargeter.ik_limits


def test_velocity_limit_applied(monkeypatch):
    calls = _spy(monkeypatch)
    retargeter = GMR("bvh_lafan1", "sprout", verbose=False, use_velocity_limit=True)
    for seed in (1, 2, 3):
        retargeter.retarget(_frame(seed))
    joint_vel = max(np.abs(vel[6:]).max() for _, _, vel in calls)  # skip the floating base
    # Before the fix the velocity limit (3*pi rad/s per joint) was dropped and steps reached ~94 rad/s here.
    assert joint_vel <= 3 * np.pi + 1e-6
