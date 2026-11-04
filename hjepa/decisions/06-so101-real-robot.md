# Decision 06: H-JEPA, JEPA-TTT and looping on a real SO-101 arm

**Date:** 2026-11-02
**Status:** planned, after the FourRoom before and after benchmarks

## Setup we have

- An assembled SO-101 **follower** arm. There's no leader arm.
- No USB camera, but several phones.
- A real local GPU for this phase. Kaggle can't talk to a USB arm.

## Decisions and reasons

**1. Start from the released DROID H-JEPA checkpoint, not from scratch.**
- DROID H-JEPA (checked in the repo config):
  - 2 levels;
  - a ViT-Small encoder at 256 px and 12-layer causal-transformer predictors;
  - 7-d actions (end-effector pose deltas plus gripper) at 5 fps;
  - two external cameras.
- It was trained from scratch on about 76k trajectories. SO-101 data is far smaller, which is enough to adapt a model but not to train a ViT world model.

**2. Map SO-101 joints to DROID's action space.**
- SO-101 records 6 joint positions.
- Forward kinematics from the public URDF turns them into the same 7-d end-effector deltas, so the DROID predictors' action input still fits.

**3. Train on three data sources, in order:**

| Data | Why |
|---|---|
| [armnet/armnetbench_v01_lerobot_so101](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101) | **Main public set.** It's the right arm: 2,499 episodes, about 15.7 h, 20 fps, 6-d joint state and action, front, top and wrist cameras, Apache-2.0. About 60% of the episodes are failed policy rollouts, which is *good* for a world model because it sees bad actions too. |
| [HuggingFaceVLA/community_dataset_v2](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v2) | **Volume and visual variety.** SO-100: 6,325 episodes, about 46.6 h, 117 contributors, Apache-2.0. It has the same 6 joints as SO-101 (from memory, not checked). |
| **Your own exploration data** | **Required.** The planner encodes goal images from *your* cameras, so the encoder must see your scene. Public data is teleoperated demos (a thin slice of actions). The planner explores bad actions too, so the model must predict them. |

No public SO-101 random-exploration dataset exists. The closest world-model projects (DreamZero-SO101, LeRobot-WAM) also used demos or simulation. The numbers above come from the dataset pages, not the files, because Hugging Face was blocked from Claude's container at the time.

**4. Collect without a leader arm.**
- **Scripted exploration:** random but safe joint targets within limits, recorded at 5 fps.
- **Optionally, moving the arm by hand:** torque off, record joint positions and video, and use the next position as the action.

**5. Use phones as cameras.**
- DroidCam or Iriun make a phone look like a normal webcam to OpenCV, which LeRobot reads.
- Two phones, fixed about 45° apart, match DROID's left and right views. Placing them front and top matches ArmNet's views.
- No wrist camera: DROID H-JEPA doesn't use one, and it adds weight to the arm.
- Keep the phones fixed between data collection and testing, since moving one is a visual shift. Lock focus and exposure.

**6. Test tasks and shifts on the arm.**
- **Tasks:** the script moves the arm to a target pose, takes goal images, resets, and the planner must return to that pose. That gives exact start and goal pairs.
- **Shifts in software:** scale one joint's gain, add a joint offset, or swap two joints' commands. They're controlled, repeatable and safe, the analogue of FourRoom's action rotation. A weight in the gripper works as a physical shift.

## Why the real arm matters for TTT

JEPA-TTT's value only shows with live interaction under changed dynamics. Offline, DROID-style evaluation can only score plans against expert paths (Fréchet fidelity). The arm gives closed-loop success under shift, which is the strongest version of our claim.

## Risks

- **Planning latency:** replanning takes seconds, so the arm pauses between moves. That's fine for reaching tasks.
- **Domain gap:** DROID (Franka arm, lab scenes) is far from SO-101 on a desk. The fine-tune has to close that gap.
