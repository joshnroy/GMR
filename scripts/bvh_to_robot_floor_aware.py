"""Retarget LAFAN1 BVH files to a robot with the floor-aware pipeline, headless (no viewer).

    # one clip
    python scripts/bvh_to_robot_floor_aware.py --bvh_file walk1_subject1.bvh --save_path out/walk1_subject1.pkl
    # a folder of clips -> <save_dir>/<clip>.pkl; existing outputs are skipped, so an interrupted run can be resumed
    python scripts/bvh_to_robot_floor_aware.py --bvh_dir lafan1/ --save_dir out/

The pickle has the same keys as scripts/bvh_to_robot.py (fps, root_pos, root_rot as xyzw, dof_pos, local_body_pos,
link_body_list) plus ``contacts`` ({"names": [...], "weight": (T, C) array}) and ``frames_corrected``.
See general_motion_retargeting/floor_aware.py for what the pipeline does.
"""

import argparse
import os
import pickle
import time
from pathlib import Path

from general_motion_retargeting.floor_aware import ROBOT_SPECS, retarget_motion
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def retarget_file(bvh_file, save_path, robot, fps):
    t0 = time.time()
    frames, human_height = load_bvh_file(str(bvh_file), format="lafan1")
    motion = retarget_motion(frames, robot, human_height, src_human="bvh_lafan1", fps=fps)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_name(save_path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(motion, f)
    os.replace(tmp, save_path)  # never leave a partial file under the final name
    return len(frames), time.time() - t0, motion["frames_corrected"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--bvh_file", type=str, help="One LAFAN1 BVH file.")
    src.add_argument("--bvh_dir", type=str, help="A folder of LAFAN1 BVH files (all *.bvh in it).")
    parser.add_argument("--save_path", type=str, help="Output .pkl (with --bvh_file).")
    parser.add_argument("--save_dir", type=str, help="Output folder (with --bvh_dir).")
    parser.add_argument("--robot", choices=sorted(ROBOT_SPECS), default="sprout")
    parser.add_argument("--motion_fps", type=int, default=30, help="Frame rate of the BVH (LAFAN1: 30).")
    parser.add_argument("--overwrite", action="store_true", help="Redo clips whose output already exists.")
    args = parser.parse_args()

    if args.bvh_file:
        if not args.save_path:
            parser.error("--bvh_file needs --save_path")
        jobs = [(Path(args.bvh_file), Path(args.save_path))]
    else:
        if not args.save_dir:
            parser.error("--bvh_dir needs --save_dir")
        jobs = [(p, Path(args.save_dir) / f"{p.stem}.pkl") for p in sorted(Path(args.bvh_dir).glob("*.bvh"))]
        if not jobs:
            parser.error(f"no .bvh files in {args.bvh_dir}")

    for i, (bvh, out) in enumerate(jobs, 1):
        tag = f"[{i}/{len(jobs)}] {bvh.stem}"
        if out.exists() and not args.overwrite:
            print(f"{tag}: exists, skipped", flush=True)
            continue
        n, secs, corrected = retarget_file(bvh, out, args.robot, args.motion_fps)
        print(f"{tag}: {n} frames in {secs:.0f}s, frames pushed out of the floor per round {corrected} -> {out}",
              flush=True)


if __name__ == "__main__":
    main()
