"""A standing LAFAN1-like skeleton (positions in m, z up) for tests that need human frames but no mocap data."""

import numpy as np

STANDING = {
    "Hips": (0.0, 0.0, 0.95),
    "Spine2": (0.0, 0.0, 1.25),
    "Neck": (0.0, 0.0, 1.5),
    "LeftUpLeg": (0.0, 0.1, 0.9),
    "RightUpLeg": (0.0, -0.1, 0.9),
    "LeftLeg": (0.0, 0.1, 0.5),
    "RightLeg": (0.0, -0.1, 0.5),
    "LeftFootMod": (0.0, 0.1, 0.08),
    "RightFootMod": (0.0, -0.1, 0.08),
    "LeftToe": (0.12, 0.1, 0.02),
    "RightToe": (0.12, -0.1, 0.02),
    "LeftArm": (0.0, 0.2, 1.4),
    "RightArm": (0.0, -0.2, 1.4),
    "LeftForeArm": (0.0, 0.45, 1.4),
    "RightForeArm": (0.0, -0.45, 1.4),
    "LeftHand": (0.0, 0.7, 1.4),
    "RightHand": (0.0, -0.7, 1.4),
}


def standing_frame(quat=(1.0, 0.0, 0.0, 0.0), offset=(0.0, 0.0, 0.0)):
    """One frame dict {joint: [position, wxyz quaternion]}, all joints with orientation ``quat``."""
    return {k: [np.array(p) + offset, np.array(quat, dtype=float)] for k, p in STANDING.items()}
