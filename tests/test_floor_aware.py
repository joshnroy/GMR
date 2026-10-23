"""Floor-aware retargeting: contact/stance weights on synthetic data, a no-data smoke test, and a LAFAN1 regression."""

import os
import pickle
from pathlib import Path

import mujoco
import numpy as np
import pytest

from general_motion_retargeting.floor_aware import (
    FloorAwareParams,
    FloorAwareRetargeting,
    contact_heights,
    contact_weights,
    get_robot_spec,
    hysteresis,
    ramp,
    retarget_motion,
    stance_intervals,
    stance_mask,
    stance_weights,
)
from skeleton import standing_frame

SPROUT = get_robot_spec("sprout")
P = FloorAwareParams()
FPS = 30


# --- weights ---------------------------------------------------------------------------------------------------------


def test_hysteresis():
    h = [0.10, 0.03, 0.05, 0.07, 0.05, 0.02, 0.059, 0.061]
    on = hysteresis(h, z_on=0.035, z_off=0.06)
    # on below z_on; stays on until above z_off; does not come back on between z_on and z_off
    assert on.tolist() == [False, True, True, False, False, True, True, False]


def test_ramp():
    on = np.r_[np.zeros(20), np.ones(20), np.zeros(20)]
    w = ramp(on, 6)
    assert np.all(w[:15] == 0) and np.allclose(w[25:35], 1) and np.all(w[-15:] == 0)
    assert np.all(np.diff(w[15:25]) >= 0) and np.all(np.diff(w[35:45]) <= 0)
    assert 0 < w[19] < 1 and 0 < w[20] < 1  # the step is spread over the window
    assert np.all((w >= 0) & (w <= 1))
    assert np.array_equal(ramp(on, 1), on)


def _foot_clip(ankle_z, toe_z, toe_x=None):
    """Frames from the standing skeleton with the left ankle and toe heights (and toe x) given per frame."""
    frames = []
    for t, (a, z) in enumerate(zip(ankle_z, toe_z)):
        f = standing_frame()
        f["LeftFootMod"][0][2] = a
        f["LeftToe"][0][2] = z
        if toe_x is not None:
            f["LeftToe"][0][0] = toe_x[t]
        frames.append(f)
    return frames


def test_foot_height_uses_toe_or_ankle():
    # On the toes: ankle high, toe on the floor. Flat: ankle at standing height, toe a bit up.
    frames = _foot_clip(ankle_z=[0.20, SPROUT.human_ankle_height + 0.01], toe_z=[0.01, 0.05])
    j = [c.human for c in SPROUT.contacts].index("LeftFootMod")
    h = contact_heights(frames, SPROUT)[:, j]
    assert h == pytest.approx([0.01, 0.01])


def test_contact_weights():
    n = 40
    toe = np.r_[np.full(10, 0.2), np.full(20, 0.0), np.full(10, 0.2)]
    names, w = contact_weights(_foot_clip(np.full(n, 0.3), toe), SPROUT, P, FPS)
    assert names == [c.human for c in SPROUT.contacts] and w.shape == (n, len(SPROUT.contacts))
    left = w[:, names.index("LeftFootMod")]
    assert np.allclose(left[15:25], 1) and left[:5].max() == 0 and left[-5:].max() == 0
    assert np.allclose(w[3:-3, names.index("RightFootMod")], 1)  # standing skeleton: right toe at 2 cm
    assert np.all(w[:, names.index("LeftHand")] == 0)


def _still(n):
    return np.zeros((n, 2))


def test_stance_needs_low_and_still():
    n = 60
    low, high = np.full(n, 0.01), np.full(n, 0.10)
    assert stance_mask(low, _still(n), P, FPS)[5:-5].all()
    assert not stance_mask(high, _still(n), P, FPS).any()
    moving = np.c_[np.arange(n) * 0.5 / FPS, np.zeros(n)]  # 0.5 m/s, above stance_speed
    assert not stance_mask(low, moving, P, FPS).any()


def test_stance_min_duration():
    n = 60
    h = np.full(n, 0.10)
    h[20:23] = 0.01  # 3 frames = 0.1 s < stance_min_s
    assert not stance_mask(h, _still(n), P, FPS).any()
    h[20:30] = 0.01  # 10 frames
    assert stance_mask(h, _still(n), P, FPS)[20:30].all()


def test_stance_gap_closing():
    n = 60
    h = np.full(n, 0.01)
    h[30:32] = 0.10  # 2-frame interruption, shorter than stance_gap_s
    on = stance_mask(h, _still(n), P, FPS)
    assert on[10:50].all()
    h[30:40] = 0.10  # long interruption: a real step
    on = stance_mask(h, _still(n), P, FPS)
    assert not on[30:40].any() and on[10:28].all() and on[42:50].all()


def test_stance_weights_and_intervals():
    n = 60
    toe = np.full(n, 0.10)
    toe[10:40] = 0.0
    w = stance_weights(_foot_clip(np.full(n, 0.3), toe), SPROUT, P, FPS)
    assert set(w) == {"LeftFootMod", "RightFootMod"}
    assert np.allclose(w["LeftFootMod"][15:35], 1) and w["LeftFootMod"][:5].max() == 0
    assert stance_intervals(w["LeftFootMod"]) == [(10, 40)]
    assert stance_intervals(np.array([1, 1, 0, 0.6, 0.4, 1])) == [(0, 2), (3, 4), (5, 6)]
    assert stance_intervals(np.zeros(5)) == []


# --- robot tables ----------------------------------------------------------------------------------------------------


def test_unsupported_robot_raises():
    with pytest.raises(ValueError, match="no body/joint table for robot 'unitree_g1'"):
        get_robot_spec("unitree_g1")
    with pytest.raises(ValueError, match="bvh_lafan1"):
        get_robot_spec("sprout", "bvh_nokov")
    with pytest.raises(ValueError):
        FloorAwareRetargeting("bvh_lafan1", "unitree_g1", verbose=False)


def test_params_frozen():
    with pytest.raises(Exception):
        P.final_tol = 0.0


# --- pipeline smoke test (no mocap data needed) ----------------------------------------------------------------------


def test_retarget_motion_synthetic_clip():
    """Standing skeleton sinking 5 cm into the floor and back: output format, contacts, nothing deep under the floor."""
    n = 45
    sink = 0.05 * np.sin(np.linspace(0, np.pi, n))
    frames = [standing_frame(quat=(0.5, 0.5, 0.5, 0.5), offset=(0.0, 0.0, -s)) for s in sink]
    motion = retarget_motion(frames, "sprout", 1.75, fps=FPS)

    for k in ("fps", "root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list", "contacts"):
        assert k in motion
    assert motion["root_pos"].shape == (n, 3) and motion["root_rot"].shape == (n, 4)
    assert motion["dof_pos"].shape == (n, 27)
    assert np.allclose(np.linalg.norm(motion["root_rot"], axis=1), 1)
    names = motion["contacts"]["names"]
    assert motion["contacts"]["weight"][:, names.index("LeftFootMod")].max() == 1

    retargeter = FloorAwareRetargeting("bvh_lafan1", "sprout", verbose=False)
    m, d = retargeter.model, mujoco.MjData(retargeter.model)
    floor = retargeter.floor
    geoms = [g for g in range(m.ngeom) if g != floor and m.geom_bodyid[g] > 0 and m.geom_contype[g]]
    worst = 0.0
    for t in range(n):
        d.qpos[:] = np.r_[motion["root_pos"][t], motion["root_rot"][t][[3, 0, 1, 2]], motion["dof_pos"][t]]
        mujoco.mj_kinematics(m, d)
        worst = min(worst, min(mujoco.mj_geomDistance(m, d, g, floor, 0.5, None) for g in geoms))
    assert worst > -P.final_tol - 0.005


# --- regression against the reference LAFAN1 -> Sprout output -------------------------------------------------------

# Point these at a LAFAN1 BVH folder and at reference outputs of the pipeline to run the regression test.
LAFAN1_BVH_DIR = Path(os.environ.get("GMR_LAFAN1_BVH_DIR", ""))
REFERENCE_DIR = Path(os.environ.get("GMR_FLOOR_AWARE_REFERENCE_DIR", ""))
REGRESSION_CLIP = "push1_subject2"  # the shortest LAFAN1 clip (3066 frames)


@pytest.mark.slow
def test_regression_lafan1_clip():
    bvh = LAFAN1_BVH_DIR / f"{REGRESSION_CLIP}.bvh"
    ref_path = REFERENCE_DIR / f"{REGRESSION_CLIP}.pkl"
    if not (os.environ.get("GMR_LAFAN1_BVH_DIR") and bvh.is_file() and ref_path.is_file()):
        pytest.skip("set GMR_LAFAN1_BVH_DIR and GMR_FLOOR_AWARE_REFERENCE_DIR to run the LAFAN1 regression test")
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, height = load_bvh_file(str(bvh), format="lafan1")
    motion = retarget_motion(frames, "sprout", height)
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)
    for k in ("root_pos", "root_rot", "dof_pos"):
        np.testing.assert_allclose(motion[k], ref[k], rtol=0, atol=1e-6, err_msg=k)
    np.testing.assert_allclose(motion["contacts"]["weight"], ref["contacts"]["weight"], rtol=0, atol=1e-9)
    assert motion["contacts"]["names"] == ref["contacts"]["names"]
