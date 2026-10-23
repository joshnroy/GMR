"""Floor-aware retargeting: GMR's per-frame IK plus floor contact, temporal smoothing and foot locking.

Plain GMR retargets each frame on its own and knows nothing about the floor. On LAFAN1 to Sprout this showed as
geometry up to 39 cm under the floor, feet, knees and hands hovering while the human touches the floor, jitter on the
ground and in dances, shoulder-twist flips in raised-arm poses, a torso twisting about twice as much as the human, and
feet sliding while planted. This module reduces those in two stages.

IK stage (``FloorAwareRetargeting``, a drop-in subclass of ``GeneralMotionRetargeting``):

* Floor constraint: a mink ``CollisionAvoidanceLimit`` keeps every robot collision geom from moving into the floor.
* Contact targets: while a human foot, knee, hand, hips or chest is on the floor (``contact_weights``, and
  ``stance_weights`` for feet), the matching robot link is pulled onto the floor below the scaled human position, and
  GMR's own target for that link is moved down too. Otherwise GMR's target, a few cm above the floor because of the
  scaling and link offsets, holds the link in the air.
* Temporal smoothing: posture and pelvis-orientation tasks pull toward the previous frame's solution.
* Torso aimed by the human chest (shoulder line and hips-to-neck line) rather than the ``Spine2`` joint, which rotates
  about 15% more than the chest; torso and pelvis orientation weighted up, which cut the torso's twist against the
  hips from about 2x the human's to 1.2x (std 8.4 -> 5.9 deg vs 4.8 for the human; dance2_subject1, 8-20 s).
* Arms aimed, not twisted: the elbow and hand target the human upper-arm and forearm directions at the robot's own
  segment lengths, and the shoulder yaw/roll are pulled toward the previous frame. With the arm raised two of Sprout's
  shoulder axes line up and GMR jumped between solutions (shoulder twist at 1,100-1,500 deg/s); frames above
  600 deg/s went from 95 to 2 on dance2_subject1, 8-20 s.

Post-processing (``postprocess``, on the whole clip):

1. Zero-phase low-pass of root and joints.
2. Push-out: each frame with geometry under the floor gets a small IK that lifts only the parts under the floor, the
   corrections are smoothed over time, and what is still more than 5 mm under is pushed once more. (The collision
   limit only stops geometry going further in; it never pushes out what is already under.)
3. Contact pinning and foot locking: one IK pass makes each contact link follow a smooth path whose height is blended
   onto the floor by its contact weight; a planted foot targets one fixed floor point (its median position over the
   stance). The pass's corrections are smoothed. GMR has no notion of a planted foot; over LAFAN1 this cut the foot's
   horizontal wander within a stance from 3.7 to 2.5 cm (median).
4. Final push-out of anything left more than 3 cm under the floor, in up to three smoothed rounds. The tolerance is
   deliberately loose: tighter tolerances lifted raised contacts back off the floor.

Only robots with an entry in ``ROBOT_SPECS`` are supported (currently Sprout, from LAFAN1 BVH).
"""

from dataclasses import dataclass
from typing import Optional

import mink
import mujoco
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

from .motion_retarget import GeneralMotionRetargeting


# ---------------------------------------------------------------------------------------------------------------------
# Robot tables


@dataclass(frozen=True)
class ContactSpec:
    """A human joint that can touch the floor and the robot link that should then rest on it."""

    human: str  # human joint (key in the frame dict and in GMR's IK tables)
    link: str  # robot body
    z_on: float  # m: human height below which the joint counts as on the floor...
    z_off: float  # ...and above which it counts as off again (hysteresis)
    toe: Optional[str] = None  # feet only: the toe joint; the foot height is min(ankle - ankle_height, toe)


@dataclass(frozen=True)
class ArmSpec:
    human_shoulder: str
    human_elbow: str
    human_wrist: str
    shoulder_body: str  # robot body at the shoulder (the arm targets start here)
    elbow_body: str
    hand_body: str


@dataclass(frozen=True)
class RobotSpec:
    """Body and joint names the floor-aware pipeline needs for one robot and one human format."""

    src_human: str  # human format the human joint names below refer to
    floor_geom: str
    pelvis_body: str
    human_ankle_height: float  # m: human ankle height when standing flat
    contacts: tuple  # of ContactSpec; feet are the entries with ``toe`` set
    human_torso: str  # human joint of GMR's torso target
    human_pelvis: str  # human joint of GMR's pelvis target
    human_chest: tuple  # human (left shoulder, right shoulder, neck, hips)
    robot_chest: tuple  # robot bodies (left shoulder, right shoulder, head, pelvis)
    arms: tuple  # of ArmSpec
    upper_arm_links: tuple  # robot links of GMR's upper-arm orientation targets
    arm_point_links: tuple  # robot links of GMR's elbow and hand position targets
    twist_joints: tuple  # shoulder joints held toward the previous frame
    arm_joints: tuple  # joints low-passed at ``arm_smooth_hz``

    @property
    def feet(self):
        return tuple(c for c in self.contacts if c.toe is not None)


ROBOT_SPECS = {
    "sprout": RobotSpec(
        src_human="bvh_lafan1",
        floor_geom="floor",
        pelvis_body="pelvis_link",
        human_ankle_height=0.065,
        contacts=(
            ContactSpec("LeftFootMod", "left_foot_link", 0.035, 0.06, toe="LeftToe"),
            ContactSpec("RightFootMod", "right_foot_link", 0.035, 0.06, toe="RightToe"),
            ContactSpec("LeftLeg", "left_tibia_link", 0.08, 0.11),
            ContactSpec("RightLeg", "right_tibia_link", 0.08, 0.11),
            ContactSpec("LeftHand", "left_gripper_lower_link", 0.08, 0.11),
            ContactSpec("RightHand", "right_gripper_lower_link", 0.08, 0.11),
            ContactSpec("Hips", "pelvis_link", 0.20, 0.24),
            ContactSpec("Spine2", "torso_link", 0.25, 0.29),
        ),
        human_torso="Spine2",
        human_pelvis="Hips",
        human_chest=("LeftArm", "RightArm", "Neck", "Hips"),
        robot_chest=("left_humerus_upper_link", "right_humerus_upper_link", "head_link", "pelvis_link"),
        arms=(
            ArmSpec("LeftArm", "LeftForeArm", "LeftHand",
                    "left_humerus_upper_link", "left_radius_link", "left_gripper_lower_link"),
            ArmSpec("RightArm", "RightForeArm", "RightHand",
                    "right_humerus_upper_link", "right_radius_link", "right_gripper_lower_link"),
        ),
        upper_arm_links=("left_humerus_lower_link", "right_humerus_lower_link"),
        arm_point_links=("left_radius_link", "right_radius_link",
                         "left_gripper_lower_link", "right_gripper_lower_link"),
        twist_joints=("left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                      "right_shoulder_roll_joint", "right_shoulder_yaw_joint"),
        arm_joints=tuple(
            f"{side}_{j}_joint" for side in ("left", "right")
            for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch")
        ),
    ),
}


def get_robot_spec(robot, src_human="bvh_lafan1"):
    if robot not in ROBOT_SPECS:
        raise ValueError(
            f"Floor-aware retargeting has no body/joint table for robot '{robot}' (supported: {sorted(ROBOT_SPECS)}). "
            "Add a RobotSpec to general_motion_retargeting.floor_aware.ROBOT_SPECS."
        )
    spec = ROBOT_SPECS[robot]
    if spec.src_human != src_human:
        raise ValueError(f"The floor-aware table for '{robot}' uses {spec.src_human} joint names, not {src_human}.")
    return spec


# ---------------------------------------------------------------------------------------------------------------------
# Parameters


@dataclass(frozen=True)
class FloorAwareParams:
    """Tuning of the floor-aware pipeline. The defaults are the values chosen on LAFAN1 -> Sprout."""

    # Contacts (feet, knees, hands, hips, chest) from the human heights in ``RobotSpec.contacts``.
    contact_ramp_s: float = 0.2  # contact weights ramp over this window instead of switching on/off
    contact_cost: float = 30.0  # IK position cost of a contact target, times its weight (GMR's foot targets use 50)
    contact_press: float = 0.01  # m: contact targets sit this far below "lowest geometry on the floor"

    # Feet: planted (stance) while the human foot is low *and* nearly still. Height-only contact released after the
    # human foot had already left the floor, so the robot foot dragged.
    stance_height: float = 0.03  # m
    stance_speed: float = 0.15  # m/s, horizontal toe speed
    stance_min_s: float = 0.15  # shorter stances are dropped
    stance_gap_s: float = 0.1  # shorter interruptions of a stance are closed
    stance_ramp_s: float = 0.1

    # IK stage.
    floor_detect_dist: float = 0.05  # m: the floor constraint acts on geoms closer than this
    smooth_joint_cost: float = 0.3  # pull of each joint toward the previous frame
    smooth_root_rot_cost: float = 3.0  # pull of the pelvis orientation toward the previous frame
    twist_hold_cost: float = 5.0  # pull of the shoulder yaw/roll toward the previous frame (20 held back real motion)
    arm_dir_cost: float = 30.0  # elbow/hand targets along the human arm directions
    upper_arm_rot_cost: float = 2.0  # GMR's upper-arm orientation (100), kept only as a tiebreak for a straight elbow
    torso_rot_cost2: float = 300.0  # torso orientation, GMR's second pass (GMR: 10)
    pelvis_rot_cost: float = 100.0  # pelvis orientation, both passes (GMR: 10 / 5)

    # Post-processing: smoothing.
    smooth_hz: float = 5.0  # joint and root-orientation low-pass
    root_smooth_hz: float = 3.0  # root position low-pass (stronger, against bounce)
    arm_smooth_hz: float = 5.0  # arm joints (6.5-8 Hz added jitter without making fast hands faster)

    # Post-processing: push-out of the floor.
    push_iters: int = 40
    push_hold_cost: float = 1.0  # pull toward the smoothed pose
    push_lift_cost: float = 50.0  # pull of a part under the floor up to just above it
    push_clearance: float = 0.002  # m above the floor that a lifted part aims for
    push_tol: float = 1e-3  # m: first push-out fixes penetration deeper than this...
    push_tol2: float = 0.005  # ...the second (after smoothing the first's corrections) only deeper than this

    # Post-processing: contact pinning and foot locking.
    pin_cost: float = 40.0  # pull of a contact link onto its pinned path, times its weight
    lock_cost: float = 60.0  # same for a planted foot onto its locked floor point
    pin_hold_cost: float = 1.0
    pin_path_hz: float = 3.0  # low-pass of the pinned target heights
    pin_iters: int = 30
    pin_min_weight: float = 0.05  # contacts below this weight are not pinned
    pin_correction_hz: float = 3.0  # low-pass of the pin pass's (and final rounds') joint corrections

    # Post-processing: final push-out.
    final_rounds: int = 3
    final_tol: float = 0.03  # m: only penetration deeper than this (tighter lifted raised contacts off the floor)


# ---------------------------------------------------------------------------------------------------------------------
# Contact and stance weights


def hysteresis(height, z_on, z_off):
    """On once ``height`` drops below ``z_on``, off again once it rises above ``z_off``."""
    on = np.zeros(len(height), dtype=bool)
    state = False
    for t, h in enumerate(height):
        state = h < z_off if state else h < z_on
        on[t] = state
    return on


def ramp(on, frames):
    """Centered moving average of a 0/1 signal, so weights ramp over ``frames`` frames instead of switching."""
    frames = max(1, int(frames))
    return np.clip(np.convolve(np.asarray(on, dtype=float), np.ones(frames) / frames, mode="same"), 0.0, 1.0)


def contact_heights(frames, spec):
    """(T, C) human height of each contact in ``spec.contacts``; feet use the lower of the toe and the ankle."""
    out = np.zeros((len(frames), len(spec.contacts)))
    for t, f in enumerate(frames):
        for j, c in enumerate(spec.contacts):
            h = f[c.human][0][2]
            if c.toe is not None:
                h = min(h - spec.human_ankle_height, f[c.toe][0][2])
            out[t, j] = h
    return out


def contact_weights(frames, spec, params=FloorAwareParams(), fps=30):
    """Per frame and contact, a weight in [0, 1]: height hysteresis, then a ramp of ``contact_ramp_s``."""
    heights = contact_heights(frames, spec)
    n = int(round(params.contact_ramp_s * fps))
    weights = np.stack(
        [ramp(hysteresis(heights[:, j], c.z_on, c.z_off), n) for j, c in enumerate(spec.contacts)], axis=1
    )
    return [c.human for c in spec.contacts], weights


def stance_mask(height, toe_xy, params=FloorAwareParams(), fps=30):
    """Planted while low and nearly still, with short gaps closed and short stances dropped."""
    speed = np.linalg.norm(np.gradient(uniform_filter1d(toe_xy, 3, axis=0), axis=0), axis=1) * fps
    on = (height < params.stance_height) & (speed < params.stance_speed)
    on = binary_closing(on, np.ones(max(1, int(params.stance_gap_s * fps)) + 1))
    return binary_opening(on, np.ones(max(1, int(params.stance_min_s * fps))))


def stance_weights(frames, spec, params=FloorAwareParams(), fps=30):
    """Per foot contact name, a stance weight in [0, 1] (``stance_mask`` ramped over ``stance_ramp_s``)."""
    heights = contact_heights(frames, spec)
    out = {}
    for j, c in enumerate(spec.contacts):
        if c.toe is None:
            continue
        toe_xy = np.array([f[c.toe][0][:2] for f in frames])
        on = stance_mask(heights[:, j], toe_xy, params, fps)
        out[c.human] = ramp(on, int(round(params.stance_ramp_s * fps)))
    return out


def stance_intervals(weight):
    """[start, end) frame ranges where ``weight`` > 0.5."""
    out, start = [], None
    for i, x in enumerate(list(np.asarray(weight) > 0.5) + [False]):
        if x and start is None:
            start = i
        elif not x and start is not None:
            out.append((start, i))
            start = None
    return out


# ---------------------------------------------------------------------------------------------------------------------
# IK stage


def _body(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _collision_geoms(model, floor):
    return [
        g for g in range(model.ngeom)
        if g != floor and model.geom_bodyid[g] > 0 and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]


def _floor_limit(model, geoms, floor, detect_dist):
    return mink.CollisionAvoidanceLimit(
        model, geom_pairs=[(geoms, [floor])], minimum_distance_from_collisions=0.0,
        collision_detection_distance=detect_dist,
    )


def _chest_frame(left_shoulder, right_shoulder, up):
    """Frame with x along the right-to-left shoulder line and z along ``up`` (made orthogonal to x)."""
    x = left_shoulder - right_shoulder
    x = x / np.linalg.norm(x)
    z = up - np.dot(up, x) * x
    z = z / np.linalg.norm(z)
    return R.from_matrix(np.stack([x, np.cross(z, x), z], axis=1))


class FloorAwareRetargeting(GeneralMotionRetargeting):
    """GMR with the floor constraint, contact targets, temporal smoothing, chest-aimed torso and direction-aimed arms.

    Call ``retarget(frame, contact_weights)`` once per frame in order, with the frame's row of ``contact_weights``
    (one weight per ``spec.contacts`` entry; zeros disables the contact targets). ``retarget_motion`` does this for a
    whole clip and runs the post-processing.
    """

    def __init__(self, src_human, tgt_robot, actual_human_height=None, params=FloorAwareParams(), **kwargs):
        self.spec = spec = get_robot_spec(tgt_robot, src_human)
        self.params = p = params
        super().__init__(src_human, tgt_robot, actual_human_height, **kwargs)
        m = self.model
        self._gmr_tasks1, self._gmr_tasks2 = list(self.tasks1), list(self.tasks2)

        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, spec.floor_geom)
        geoms = _collision_geoms(m, self.floor)
        self.ik_limits.append(_floor_limit(m, geoms, self.floor, p.floor_detect_dist))

        # Contact targets (cost 0 until a contact's weight is positive).
        self.contact_tasks, self.link_geoms = [], []
        for c in spec.contacts:
            task = mink.FrameTask(c.link, "body", position_cost=0.0, orientation_cost=0.0, lm_damping=1)
            task.set_target(mink.SE3.identity())
            self.contact_tasks.append(task)
            b = _body(m, c.link)
            self.link_geoms.append([g for g in geoms if m.geom_bodyid[g] == b])

        # Temporal smoothing toward the previous frame (off on the first frame).
        self.posture_task = mink.PostureTask(m, cost=0.0)
        self.root_smooth_task = mink.FrameTask(spec.pelvis_body, "body", position_cost=0.0, orientation_cost=0.0,
                                               lm_damping=1)
        self.twist_cost = np.zeros(m.nv)
        for name in spec.twist_joints:
            self.twist_cost[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = p.twist_hold_cost
        self.twist_task = mink.PostureTask(m, cost=np.zeros(m.nv))

        # Reweight GMR's own targets: arms are aimed by direction (below); torso and pelvis held to the human.
        for task in self._gmr_tasks1 + self._gmr_tasks2:
            if task.frame_name in spec.upper_arm_links:
                task.set_orientation_cost(p.upper_arm_rot_cost)
                task.set_position_cost(0.0)
            elif task.frame_name in spec.arm_point_links:
                task.set_position_cost(0.0)
        if spec.human_torso in self.human_body_to_task2:
            self.human_body_to_task2[spec.human_torso].set_orientation_cost(p.torso_rot_cost2)
        for table in (self.human_body_to_task1, self.human_body_to_task2):
            if spec.human_pelvis in table:
                table[spec.human_pelvis].set_orientation_cost(p.pelvis_rot_cost)

        # Arm direction targets at the robot's own segment lengths (zero pose).
        d0 = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d0)
        self.arms = []
        for a in spec.arms:
            sh, el, wr = (_body(m, n) for n in (a.shoulder_body, a.elbow_body, a.hand_body))
            self.arms.append({
                "spec": a,
                "shoulder_body": sh,
                "upper": float(np.linalg.norm(d0.xpos[el] - d0.xpos[sh])),
                "fore": float(np.linalg.norm(d0.xpos[wr] - d0.xpos[el])),
                "elbow_task": mink.FrameTask(a.elbow_body, "body", position_cost=p.arm_dir_cost, orientation_cost=0.0,
                                             lm_damping=1),
                "hand_task": mink.FrameTask(a.hand_body, "body", position_cost=p.arm_dir_cost, orientation_cost=0.0,
                                            lm_damping=1),
            })

        # Fixed rotation from the robot's chest frame (shoulder line, pelvis-to-head line) to its torso frame.
        ls, rs, head, pelvis = (d0.xpos[_body(m, n)] for n in spec.robot_chest)
        torso_link = self.human_body_to_task1[spec.human_torso].frame_name
        self.chest_to_torso = _chest_frame(ls, rs, head - pelvis).inv() * R.from_matrix(
            d0.xmat[_body(m, torso_link)].reshape(3, 3))

        extra = self.contact_tasks + [self.posture_task, self.root_smooth_task, self.twist_task]
        extra += [arm[k] for arm in self.arms for k in ("elbow_task", "hand_task")]
        self.tasks1 = self.tasks1 + extra
        self.tasks2 = self.tasks2 + extra
        self._first_frame = True
        self._weights = np.zeros(len(spec.contacts))

    def retarget(self, human_data, contact_weights=None, offset_to_ground=False):
        self._weights = np.zeros(len(self.spec.contacts)) if contact_weights is None else np.asarray(contact_weights)
        return super().retarget(human_data, offset_to_ground)

    # GMR iterates each frame until its own matching error stops improving; leave the added terms out of that error.
    def error1(self):
        return np.linalg.norm(np.concatenate([t.compute_error(self.configuration) for t in self._gmr_tasks1]))

    def error2(self):
        return np.linalg.norm(np.concatenate([t.compute_error(self.configuration) for t in self._gmr_tasks2]))

    def _floor_contact_height(self, j):
        """Height for contact ``j``'s link origin that puts its lowest geometry ``contact_press`` below the floor."""
        d = self.configuration.data
        gap = min(mujoco.mj_geomDistance(self.model, d, g, self.floor, 1.0, None) for g in self.link_geoms[j])
        origin_z = self.configuration.get_transform_frame_to_world(self.spec.contacts[j].link, "body").translation()[2]
        return origin_z - gap - self.params.contact_press

    def update_targets(self, human_data, offset_to_ground=False):
        super().update_targets(human_data, offset_to_ground)
        spec, p = self.spec, self.params
        tables = (self.human_body_to_task1, self.human_body_to_task2)
        data = self.configuration.data
        mujoco.mj_forward(self.model, data)  # the configuration still holds the previous frame's solution

        on = 0.0 if self._first_frame else 1.0
        self.posture_task.set_target_from_configuration(self.configuration)
        self.posture_task.set_cost(p.smooth_joint_cost * on)
        self.root_smooth_task.set_target_from_configuration(self.configuration)
        self.root_smooth_task.set_orientation_cost(p.smooth_root_rot_cost * on)
        self.twist_task.set_target_from_configuration(self.configuration)
        if not self._first_frame:
            self.twist_task.set_cost(self.twist_cost)
        self._first_frame = False

        # Contacts: a target on the floor under the scaled human position, and GMR's own targets on that link moved
        # down to the floor by the contact weight.
        for j, c in enumerate(spec.contacts):
            w = self._weights[j]
            task = self.contact_tasks[j]
            if w <= 0:
                task.set_position_cost(0.0)
                continue
            z_floor = self._floor_contact_height(j)
            xy = np.asarray(self.scaled_human_data[c.human][0])[:2]
            task.set_target(mink.SE3.from_translation(np.array([xy[0], xy[1], z_floor])))
            task.set_position_cost(p.contact_cost * w)
            for table in tables:
                gmr_task = table.get(c.human)
                if gmr_task is None or gmr_task.transform_target_to_world is None:
                    continue
                T = gmr_task.transform_target_to_world
                pos = T.translation().copy()
                pos[2] = (1 - w) * pos[2] + w * z_floor
                gmr_task.set_target(mink.SE3.from_rotation_and_translation(T.rotation(), pos))

        # Torso: shoulder line and up axis from the human chest instead of the Spine2 joint.
        if all(k in human_data for k in spec.human_chest):
            ls, rs, neck, hips = (np.asarray(human_data[k][0]) for k in spec.human_chest)
            rot = mink.SO3((_chest_frame(ls, rs, neck - hips) * self.chest_to_torso).as_quat(scalar_first=True))
            for table in tables:
                task = table.get(spec.human_torso)
                if task is None or task.transform_target_to_world is None:
                    continue
                pos = task.transform_target_to_world.translation()
                task.set_target(mink.SE3.from_rotation_and_translation(rot, pos))

        # Arms: elbow and hand along the human arm directions, from the robot's current shoulder.
        for arm in self.arms:
            a = arm["spec"]
            sh_h, el_h, wr_h = (np.asarray(human_data[k][0]) for k in (a.human_shoulder, a.human_elbow, a.human_wrist))
            upper, fore = el_h - sh_h, wr_h - el_h
            elbow = data.xpos[arm["shoulder_body"]] + arm["upper"] * upper / np.linalg.norm(upper)
            hand = elbow + arm["fore"] * fore / np.linalg.norm(fore)
            arm["elbow_task"].set_target(mink.SE3.from_translation(elbow))
            arm["hand_task"].set_target(mink.SE3.from_translation(hand))


# ---------------------------------------------------------------------------------------------------------------------
# Post-processing


def lowpass(x, fps, hz):
    """Zero-phase 4th-order Butterworth low-pass along time (axis 0)."""
    b, a = butter(4, hz / (fps / 2))
    return filtfilt(b, a, x, axis=0)


def smooth_quat(q, fps, hz):
    """Low-pass a quaternion track (sign-aligned frame to frame first), renormalized."""
    q = q.copy()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    q = lowpass(q, fps, hz)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def smooth_joints(dof, fps, hz, arm_idx=(), arm_hz=None):
    """Low-pass joint angles at ``hz``, with the ``arm_idx`` columns at ``arm_hz``."""
    out = lowpass(dof, fps, hz)
    if len(arm_idx):
        out[:, arm_idx] = lowpass(dof[:, arm_idx], fps, arm_hz)
    return out


def _qpos(root_pos, root_rot_xyzw, dof):
    return np.concatenate([root_pos, root_rot_xyzw[..., [3, 0, 1, 2]], dof], axis=-1)


class FloorProjector:
    """Per-frame IK that lifts the parts of a pose that are under the floor while holding the rest of the pose."""

    def __init__(self, model, spec, params=FloorAwareParams()):
        self.model, self.params = model, params
        self.config = mink.Configuration(model)
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, spec.floor_geom)
        self.geoms = _collision_geoms(model, self.floor)
        self.limits = [mink.ConfigurationLimit(model),
                       _floor_limit(model, self.geoms, self.floor, params.floor_detect_dist)]
        self.hold = mink.PostureTask(model, cost=params.push_hold_cost)
        # The posture task leaves the floating base free; hold the root pose too.
        self.hold_root = mink.FrameTask(spec.pelvis_body, "body", position_cost=params.push_hold_cost,
                                        orientation_cost=params.push_hold_cost, lm_damping=1)
        bodies = sorted({int(model.geom_bodyid[g]) for g in self.geoms})
        self.lift = {
            b: mink.FrameTask(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b), "body", position_cost=0.0,
                              orientation_cost=0.0, lm_damping=1)
            for b in bodies
        }

    def body_gaps(self):
        """Per robot body: its smallest geom-to-floor distance in the current configuration (negative: under)."""
        d, out = self.config.data, {}
        for g in self.geoms:
            b = int(self.model.geom_bodyid[g])
            out[b] = min(out.get(b, np.inf), mujoco.mj_geomDistance(self.model, d, g, self.floor, 0.5, None))
        return out

    def _fixes(self, tol):
        return {b: -g + self.params.push_clearance for b, g in self.body_gaps().items() if g < -tol}

    def project(self, qpos, tol):
        """Lift every body more than ``tol`` under the floor. Returns the new qpos and the IK steps taken (0: none)."""
        p = self.params
        self.config.update(qpos)
        fixes = self._fixes(tol)
        if not fixes:
            return qpos, 0
        self.hold.set_target(qpos)
        self.hold_root.set_target_from_configuration(self.config)
        dt = self.model.opt.timestep
        for it in range(p.push_iters):
            tasks = [self.hold, self.hold_root]
            for b, dz in fixes.items():
                pos = self.config.data.xpos[b].copy()
                pos[2] += dz
                self.lift[b].set_target(mink.SE3.from_translation(pos))
                self.lift[b].set_position_cost(p.push_lift_cost)
                tasks.append(self.lift[b])
            vel = mink.solve_ik(self.config, tasks, dt, "daqp", 1e-3, limits=self.limits)
            self.config.integrate_inplace(vel, dt)
            fixes = self._fixes(tol)
            if not fixes:
                break
        return self.config.q.copy(), it + 1


def push_out_of_floor(root_pos, root_rot, dof, projector, tol):
    """Project every frame out of the floor, in place (root position and joints; the smoothed root orientation is
    kept). Returns the number of frames changed."""
    n = 0
    for t in range(len(dof)):
        q, steps = projector.project(_qpos(root_pos[t], root_rot[t], dof[t]), tol)
        if steps:
            n += 1
            root_pos[t], dof[t] = q[:3], q[7:]
    return n


def pin_contacts(root_pos, root_rot, dof, model, spec, weights, foot_weights, params=FloorAwareParams(), fps=30):
    """Make each contact link follow a smooth floor-pinned path, and each planted foot a fixed floor point.

    The target path of a contact link is its current path with the height blended onto the floor by the contact
    weight, low-passed so following it adds no vibration (earlier pull-downs that reacted to each frame's gap did).
    During a stance the foot's horizontal target is its median position over the stance, blended in and out by the
    stance weight. Returns new (root_pos, root_rot, dof).
    """
    p = params
    proj = FloorProjector(model, spec, p)
    links = [_body(model, c.link) for c in spec.contacts]
    q_all = _qpos(root_pos, root_rot, dof)
    T, C = len(q_all), len(links)

    pos, gap = np.zeros((T, C, 3)), np.zeros((T, C))
    for t in range(T):
        proj.config.update(q_all[t])
        gaps = proj.body_gaps()
        for j, b in enumerate(links):
            pos[t, j] = proj.config.data.xpos[b]
            gap[t, j] = gaps.get(b, 0.0)
    targets = pos.copy()
    targets[:, :, 2] = lowpass(pos[:, :, 2] - weights * (gap - p.contact_press), fps, p.pin_path_hz)

    cost_w = weights.copy()
    names = [c.human for c in spec.contacts]
    pad = max(1, int(round(p.stance_ramp_s * fps)))  # include the ramp frames around each stance
    for k, lw in foot_weights.items():
        j = names.index(k)
        for a, b in stance_intervals(lw):
            lock = np.median(pos[a:b, j, :2], axis=0)
            a0, b0 = max(0, a - pad), min(T, b + pad)
            targets[a0:b0, j, :2] = (1 - lw[a0:b0, None]) * pos[a0:b0, j, :2] + lw[a0:b0, None] * lock
        cost_w[:, j] = np.maximum(weights[:, j], lw)

    pin_tasks = [mink.FrameTask(c.link, "body", position_cost=0.0, orientation_cost=0.0, lm_damping=1)
                 for c in spec.contacts]
    link_cost = [p.lock_cost if c.toe is not None else p.pin_cost for c in spec.contacts]
    proj.hold.set_cost(p.pin_hold_cost)
    dt = model.opt.timestep
    out = q_all.copy()
    q_prev = q_all[0]
    for t in range(T):
        active = [j for j in range(C) if cost_w[t, j] > p.pin_min_weight]
        if not active:
            q_prev = q_all[t]
            continue
        proj.config.update(q_all[t])
        proj.hold.set_target(q_all[t])
        proj.hold_root.set_target_from_configuration(proj.config)
        proj.config.update(q_prev)  # warm start from the previous result
        tasks = [proj.hold, proj.hold_root]
        for j in active:
            pin_tasks[j].set_target(mink.SE3.from_translation(targets[t, j]))
            pin_tasks[j].set_position_cost(link_cost[j] * cost_w[t, j])
            tasks.append(pin_tasks[j])
        for _ in range(p.pin_iters):
            vel = mink.solve_ik(proj.config, tasks, dt, "daqp", 1e-3, limits=proj.limits)
            proj.config.integrate_inplace(vel, dt)
            if np.linalg.norm(vel) * dt < 1e-4:
                break
        out[t] = proj.config.q
        q_prev = out[t]
    return out[:, :3], out[:, [4, 5, 6, 3]], out[:, 7:]


def postprocess(motion, model, spec, weights, foot_weights, params=FloorAwareParams()):
    """Smooth, push out of the floor, pin contacts and lock feet, then final push-out (see the module docstring).

    ``weights`` is the (T, C) contact weight array used in the IK (feet: stance weights), ``foot_weights`` the stance
    weight per foot contact name. Returns a copy of ``motion`` with new root_pos, root_rot and dof_pos and a
    ``frames_corrected`` list (frames changed by each push-out round).
    """
    p, fps = params, motion["fps"]
    hinge_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)
                   if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    arm_idx = [hinge_names.index(n) for n in spec.arm_joints]

    def smooth_corrections(rp, rp0, dof, dof0, joint_hz, arm=True):
        rp = rp0 + lowpass(rp - rp0, fps, p.root_smooth_hz)
        dof = dof0 + smooth_joints(dof - dof0, fps, joint_hz, arm_idx if arm else (), p.arm_smooth_hz)
        return rp, dof

    # 1. Smooth.
    rp = lowpass(motion["root_pos"], fps, p.root_smooth_hz)
    rr = smooth_quat(motion["root_rot"], fps, p.smooth_hz)
    dof = smooth_joints(motion["dof_pos"], fps, p.smooth_hz, arm_idx, p.arm_smooth_hz)

    # 2. Push out of the floor; frame-by-frame fixes jump between neighbouring frames, so smooth them and push once
    #    more for what is still deeper than push_tol2.
    proj = FloorProjector(model, spec, p)
    rp0, dof0 = rp.copy(), dof.copy()
    counts = [push_out_of_floor(rp, rr, dof, proj, p.push_tol)]
    rp, dof = smooth_corrections(rp, rp0, dof, dof0, p.smooth_hz)
    counts.append(push_out_of_floor(rp, rr, dof, proj, p.push_tol2))

    # 3. Pin contacts and lock feet; smooth the pass's corrections (the per-frame IK does not settle identically frame
    #    to frame, which showed as leg vibration on the ground).
    rp0, dof0 = rp.copy(), dof.copy()
    rp, rr, dof = pin_contacts(rp, rr, dof, model, spec, weights, foot_weights, p, fps)
    rp, dof = smooth_corrections(rp, rp0, dof, dof0, p.pin_correction_hz, arm=False)

    # 4. Final push-out of deep penetration, in smoothed rounds.
    proj = FloorProjector(model, spec, p)
    for _ in range(p.final_rounds):
        rp0, dof0 = rp.copy(), dof.copy()
        n = push_out_of_floor(rp, rr, dof, proj, p.final_tol)
        counts.append(n)
        if not n:
            break
        rp, dof = smooth_corrections(rp, rp0, dof, dof0, p.pin_correction_hz, arm=False)

    out = dict(motion)
    out.update(root_pos=rp, root_rot=rr, dof_pos=dof, frames_corrected=counts)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Whole clip


def retarget_motion(frames, tgt_robot, actual_human_height=None, src_human="bvh_lafan1", fps=30,
                    params=FloorAwareParams(), verbose=False):
    """Retarget a list of human frames (as from ``utils.lafan1.load_bvh_file``) with the floor-aware pipeline.

    Returns GMR's motion dict (root_pos, root_rot as xyzw, dof_pos, ...) plus ``contacts`` (names and per-frame
    weights) and ``frames_corrected``.
    """
    spec = get_robot_spec(tgt_robot, src_human)
    names, weights = contact_weights(frames, spec, params, fps)
    foot_weights = stance_weights(frames, spec, params, fps)
    for k, w in foot_weights.items():  # feet: stance (low and still) instead of height-only contact, everywhere
        weights[:, names.index(k)] = w

    retargeter = FloorAwareRetargeting(src_human, tgt_robot, actual_human_height, params=params, verbose=verbose)
    q = np.array([retargeter.retarget(f, weights[t]) for t, f in enumerate(frames)])
    motion = {
        "fps": fps,
        "root_pos": q[:, :3],
        "root_rot": q[:, [4, 5, 6, 3]],  # wxyz -> xyzw, as GMR's scripts save it
        "dof_pos": q[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
        "contacts": {"names": names, "weight": weights},
    }
    return postprocess(motion, retargeter.model, spec, weights, foot_weights, params)
