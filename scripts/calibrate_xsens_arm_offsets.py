"""Calibrate per-bone arm rotation offsets for retargeting an Xsens BVH skeleton
to a humanoid robot, and (optionally) write them into the IK config.

WHAT IS "offset" HERE?
----------------------
In GMR the IK orientation target for a robot link is computed in
`motion_retarget.py:offset_human_data` as

    target_rot = R_human_global(bone) * rot_offset[bone]      # right-multiply, bone-local

where `rot_offset` is the 5th element of each `ik_match_table` entry. Because every
`pos_offset` (the 4th element) is [0,0,0], changing `rot_offset` rotates ONLY the
orientation target and leaves the position target untouched. So `rot_offset` is a
constant, per-bone redefinition of the bone's local frame -- exactly the right lever
to absorb a *skeleton convention* difference (two BVH exporters defining a bone's
local axes differently) without moving any target position or touching task weights.

Different Xsens exports do NOT share one convention: the EMM export rotates the
shoulder frames ~82 deg (L) / ~87 deg (R) about world-x relative to a stock 3DSMax
export. This script measures that per-bone rotation from the rest pose and emits the
`rot_offset` that cancels it.

HOW THE OFFSET IS COMPUTED
--------------------------
We want each EMM arm bone, at the arms-down rest pose, to produce the SAME IK target
as a known-good reference. Everything is expressed relative to the pelvis so the two
clips' root headings (yaw) cancel:

    R_emm_rel  = R_pelvis_emm^-1 * R_bone_emm          # EMM bone, pelvis-relative
    R_ref_rel  = R_pelvis_ref^-1 * R_bone_ref          # reference bone, pelvis-relative
    offset[bone] = R_emm_rel^-1 * R_ref_rel * stock_offset[bone]

`stock_offset` is the rot_offset the reference clip already used (identity for the
shoulder, R_y(90) for elbow/wrist in the stock GMR bvh_xsens config), so the
known-good *target* is transferred onto the EMM bone frame.

Two reference choices are reported for cross-checking:
  * robot-neutral : reference = the robot's own qpos0 arm-link orientation (no
                    stock_offset needed; offset = R_emm_rel^-1 * R_link_rel).
  * boxing-ref    : reference = a known-good 3DSMax Xsens clip, carried via stock_offset.
The two agree on the shoulders to ~2 deg (robust); they disagree on elbow/wrist,
which are pose-dependent and unreliable (wrists are pinned to neutral post-IK anyway).
The boxing-ref values are the ones written with --write.

USAGE
-----
    # just report (no file changes):
    python scripts/calibrate_xsens_arm_offsets.py
    # compute and write boxing-ref offsets into the IK config, and zero offsets.json:
    python scripts/calibrate_xsens_arm_offsets.py --write --zero-offsets
"""
import os
import json
import types
import argparse
import pathlib
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_EMM = "/home/jkim3662/Projects/Environment-aware-Motion-Matching/DataEMM/emm/basic_locomotion_1.bvh"
DEFAULT_REF = str(REPO / "assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh")

RY90 = [0.7071067811865476, 0.0, 0.7071067811865475, 0.0]   # stock elbow/wrist offset
STOCK = {  # rot_offsets the reference (stock GMR bvh_xsens) config used
    "LeftShoulder": [1, 0, 0, 0], "RightShoulder": [1, 0, 0, 0],
    "LeftElbow": RY90, "RightElbow": RY90, "LeftWrist": RY90, "RightWrist": RY90,
}
BONE2LINK = {
    "LeftShoulder":  "left_shoulder_yaw_link",
    "RightShoulder": "right_shoulder_yaw_link",
    "LeftElbow":     "left_elbow_link",
    "RightElbow":    "right_elbow_link",
    "LeftWrist":     "left_wrist_roll_link",
    "RightWrist":    "right_wrist_roll_link",
}


def w(q):  # wxyz -> scipy Rotation
    return R.from_quat([q[1], q[2], q[3], q[0]])


def quat_wxyz(rot, ndigits=8):  # scipy Rotation -> normalized wxyz list
    q = rot.as_quat()  # xyzw
    a = np.array([q[3], q[0], q[1], q[2]], float)
    a /= np.linalg.norm(a)
    return [round(float(x), ndigits) for x in a]


def load_rest(path, scale):
    """Load a BVH and return (frame0 dict, height). Reads offsets.json from CWD."""
    from general_motion_retargeting.utils.xsens import load_xsens_file
    args = types.SimpleNamespace(bvh_file=path, scale=scale, start=None, end=None,
                                 reset_to_zero=False, bvh_format="3DSM")
    frames, height, _ = load_xsens_file(args)
    return frames[0], height


def main():
    import general_motion_retargeting.params as params

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emm_bvh", default=DEFAULT_EMM, help="BVH to calibrate (the new convention).")
    ap.add_argument("--ref_bvh", default=DEFAULT_REF, help="Known-good reference BVH (stock convention).")
    ap.add_argument("--emm_scale", type=float, default=1.0, help="EMM BVH unit scale (metres=1.0).")
    ap.add_argument("--ref_scale", type=float, default=0.01, help="Reference BVH unit scale (cm=0.01).")
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--src_human", default="bvh_xsens_emm",
                    help="IK_CONFIG_DICT key whose config receives the offsets with --write.")
    ap.add_argument("--write", action="store_true", help="Write boxing-ref offsets into the IK config.")
    ap.add_argument("--zero-offsets", action="store_true", dest="zero_offsets",
                    help="Also zero offsets.json (the calibration assumes it is all-zero).")
    args = ap.parse_args()

    # The xsens loader reads ./offsets.json relative to CWD; pin CWD to the repo so it
    # resolves and so we can neutralize it during the rest-pose load.
    os.chdir(REPO)
    offsets_path = REPO / "offsets.json"

    # --- load both rest poses with offsets.json temporarily zeroed (non-destructive) --
    orig = offsets_path.read_text() if offsets_path.exists() else None
    if orig is not None:
        zeroed = json.loads(orig)
        for k in zeroed:
            for a in ("X", "Y", "Z"):
                zeroed[k][a] = 0.0
        offsets_path.write_text(json.dumps(zeroed, indent=4))
    try:
        f_emm, height = load_rest(args.emm_bvh, args.emm_scale)
        f_ref, _ = load_rest(args.ref_bvh, args.ref_scale)
    finally:
        if orig is not None and not args.zero_offsets:
            offsets_path.write_text(orig)   # restore unless we were asked to zero it

    Rp_emm = w(f_emm["Hips"][1])
    Rp_ref = w(f_ref["Hips"][1])

    # --- robot neutral pose (qpos0) ---------------------------------------------------
    m = mj.MjModel.from_xml_path(str(params.ROBOT_XML_DICT[args.robot]))
    d = mj.MjData(m)
    mj.mj_resetData(m, d)
    mj.mj_forward(m, d)

    def link_R(name):
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"no body {name} in {args.robot}"
        return w(d.xquat[bid])
    Rp_rob = link_R("pelvis")

    def report(rot, tag):
        rv = rot.as_rotvec()
        ang = float(np.degrees(np.linalg.norm(rv)))
        ax = (rv / (np.linalg.norm(rv) + 1e-12)).round(2).tolist()
        wxyz = quat_wxyz(rot)
        print(f"    {tag:14s} {str(wxyz):46s} {ang:6.1f} deg  axis={ax}")
        return wxyz

    print(f"\nEMM rest loaded (height={height:.3f} m), robot={args.robot}\n")
    boxing_ref = {}
    for bone, link in BONE2LINK.items():
        print(f"{bone} -> {link}")
        R_emm_rel = Rp_emm.inv() * w(f_emm[bone][1])
        # reference A: robot neutral
        report(R_emm_rel.inv() * (Rp_rob.inv() * link_R(link)), "robot-neutral")
        # reference B: known-good reference clip carried via stock offset (written)
        R_ref_rel = Rp_ref.inv() * w(f_ref[bone][1])
        boxing_ref[link] = report(R_emm_rel.inv() * R_ref_rel * w(STOCK[bone]), "boxing-ref")

    print("\n=== boxing-ref rot_offsets (link -> wxyz) ===")
    print(json.dumps(boxing_ref, indent=2))

    if args.write:
        cfg_path = pathlib.Path(params.IK_CONFIG_DICT[args.src_human][args.robot])
        cfg = json.loads(cfg_path.read_text())
        for table in ("ik_match_table1", "ik_match_table2"):
            for link, q in boxing_ref.items():
                assert link in cfg[table], f"{link} missing in {table} of {cfg_path}"
                cfg[table][link][4] = q
        cfg_path.write_text(json.dumps(cfg, indent=4))
        print(f"\n[write] patched {cfg_path}")
        if args.zero_offsets and orig is not None:
            print(f"[write] zeroed {offsets_path}")


if __name__ == "__main__":
    main()
