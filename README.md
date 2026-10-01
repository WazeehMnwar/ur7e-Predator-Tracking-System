# Predator — UR7e Vision Tracking System

> Built for the **Robotics X Nature Challenge** for the Ingram Hall Makerspace at Texas State University — a real-time human tracking system for a Universal Robots UR7e arm.

The arm autonomously points at a selected person using a vision pipeline that combines YOLO body tracking, persistent face recognition, hand gesture detection, and a visual servoing controller with forward/back reach control and autonomous lost-target search.

[![Predator Tracking Demo](https://img.youtube.com/vi/Tso7jccfZYE/0.jpg)](https://youtube.com/shorts/Tso7jccfZYE)

---

## Built For

This system was developed as part of the **Robotics X Nature Challenge**, where the goal was to build a robotic system capable of autonomously perceiving and responding to humans in a natural environment. Predator handles the full pipeline — from raw camera input to physical arm motion — with no human in the loop once a target is selected.

The codebase is fully integrated and tested on a physical UR7e arm with ROS 2 Humble, live camera input, and full MoveIt Servo integration.

---

## Features

- **Person tracking** — YOLO26n + BoT-SORT multi-person detection and tracking
- **Persistent face identity** — YuNet face detection + SFace recognition; automatically enrolls and re-identifies people across sessions
- **Role system** — assign one **TARGET** (arm tracks them) and one **BOSS** (immune to tracking; their gesture controls the arm)
- **Boss gesture** — BOSS holds an open palm toward the camera for 0.5 s → arm performs a full bow sequence and returns
- **Analytic planar IK** — 4-joint closed-form IK (shoulder pan, lift, elbow, wrist 1) derived directly from the URDF. Wrist 2 and wrist 3 are deliberately locked at 90°/−90°: this keeps the camera stable and makes the arm's tracking motion visually expressive — the whole arm body sweeps and tilts rather than hiding the movement in small wrist adjustments
- **Forward/back reach** — arm moves closer or farther from the person based on their shoulder width in frame
- **Lost-target search** — if the target disappears, the arm sweeps left/right looking for them before returning to start
- **Camera calibration tool** — built-in interactive mount calibration; records manual arm poses and fits camera orientation + FOV using the URDF kinematic chain
- **Body fallback** — when the face is temporarily not visible, approximates head location from the body bounding box

---

## Architecture

```
Camera
  │
  ▼
src/vision/target_tracker.py
  │  YOLO26n body tracking (BoT-SORT)
  │  YuNet face detection
  │  SFace face recognition + persistent identity
  │  MediaPipe hand gesture (open palm → BOW)
  │  MediaPipe pose (shoulder width for reach)
  │
  │  UDP  ──  TRACK,ex,ey,age_ms
  │           BOW,age_ms
  │           BOX,height,id,valid,age_ms
  │           SHOULDERS,width,id,valid,age_ms
  │           TARGET_MISSING,age_ms
  │           START / STOP / TARGET_LOST
  ▼
src/control/vision_moveit_servo.py   (ROS 2 node)
  │  Analytic planar IK  (planar_tracking.py)
  │  Whole-arm Jacobian  (whole_arm_tracking.py)
  │  Reach control       (reach_control.py)
  │  Lost-target search  (target_search.py)
  │
  │  ROS 2
  │  JointJog → /servo_node/delta_joint_cmds → MoveIt Servo → UR driver
  ▼
Physical UR7e arm
```

---

## Requirements

### Vision machine (Jetson or any Linux/Mac with camera)
- Python 3.10+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (`pip install ultralytics`)
- OpenCV with contrib (`pip install opencv-contrib-python`)
- MediaPipe (`pip install mediapipe`)
- NumPy

### Arm machine
- Ubuntu 22.04
- ROS 2 Humble
- [Universal Robots ROS2 Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)
- [MoveIt 2](https://moveit.ros.org/) with `moveit_servo`
- `ur_moveit_config` package

Both machines can be the same machine if running locally.

---

## Model Downloads

Model files are not included in this repo (too large for Git). Download them and place them in `models/`:

| Model | Destination | Download |
|-------|-------------|----------|
| YOLO26n | `models/yolo/yolo26n.pt` | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| YuNet face detector | `models/face/face_detection_yunet_2023mar.onnx` | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| SFace face recognizer | `models/face/face_recognition_sface_2021dec.onnx` | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |
| MediaPipe Hand Landmarker | `models/hand/hand_landmarker.task` | [MediaPipe](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |
| MediaPipe Pose Landmarker (lite) | `models/hand/pose_landmarker_lite.task` | [MediaPipe](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task) |

> **Note:** The Pose Landmarker is only needed for shoulder-width reach control. If it's missing, the system falls back gracefully — pointing and face tracking still work.

---

## Setup

### 1. Clone
```bash
git clone https://github.com/WazeehMnwar/ur7e-Predator-Tracking-System.git
cd ur7e-Predator-Tracking-System
```

### 2. Download models
Place all model files in `models/` as shown in the table above.

### 3. Configure the arm
Edit `config/predator_servo_poses.yaml`:

- **`start`** — set these to your actual UR7e resting joint angles (radians) in the order: `shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3`
- **`ik.pan_sign` / `ik.tilt_sign`** — set to `+1` or `-1` after verifying which direction the image shifts when each joint moves
- **`ik.pan_span` / `ik.pitch_span`** — reduce if cable routing or workspace limits need it
- **`reach`** section — `radius_min` and `radius_max` come from FK at your near/far target poses; measure these on your hardware

### 4. Start the UR driver
```bash
# On the arm machine — standard UR ROS 2 driver startup
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur7e robot_ip:=<YOUR_ROBOT_IP> use_fake_hardware:=false
```

### 5. Launch MoveIt Servo
```bash
ros2 launch ./launch/predator_servo.launch.py
```

### 6. Start the arm bridge
```bash
# Dry run (no motion — safe for first test)
python3 src/control/vision_moveit_servo.py

# With motion enabled
python3 src/control/vision_moveit_servo.py --enable-motion

# With motion + forward/back reach
python3 src/control/vision_moveit_servo.py --enable-motion --enable-reach
```

### 7. Start vision
```bash
python3 src/vision/target_tracker.py
```

---

## Controls (Vision Window)

| Input | Action |
|-------|--------|
| **Click** a person | Select them |
| **`t`** | Make selected person the **TARGET** — arm tracks them |
| **`b`** | Make selected person the **BOSS** — arm never tracks them; their open palm triggers a bow |
| **Click empty area** or **`ESC`** | Clear selection |
| **BOSS open palm** (held 0.5 s) | Arm bows, then returns to tracking pose |
| **`q`** | Quit |

---

## Boss Bow Sequence

One of the more distinctive behaviors. When the designated BOSS person holds an open palm facing the camera for at least 0.5 seconds, the arm executes a full bow and returns to its previous pose:

1. **Detection** — MediaPipe Hand Landmarker detects the hand in real time. `boss_gesture.py` validates it's a genuine open palm: correct facing direction, all five fingers fully extended, palm close enough to the camera to be deliberate
2. **Identity check** — the wrist and palm center are spatially matched against the BOSS person's bounding box to confirm it's actually them and not someone nearby
3. **Hold timer** — must be continuously detected for 0.5 s across at least 4 frames, with a 5 s cooldown between bows to prevent accidental triggers
4. **BRAKE** — the arm decelerates to a full stop over 0.3 s
5. **LOWER** — only three joints move (`shoulder_lift`, `elbow`, `wrist_1`) to the configured bow angles; the other three joints hold their live positions so the arm doesn't snap to a fixed orientation
6. **HOLD** — holds the bow pose for 1 second
7. **RETURN** — all six joints return to the exact pre-bow pose captured at step 4

Tracking resumes automatically after the return.

---

## Camera Calibration

The calibration tool fits the camera's physical mounting orientation and offset relative to the `tool0` link, using the URDF-derived arm kinematics. Run it before deploying to a new physical setup:

```bash
python3 src/control/vision_moveit_servo.py --calibrate-camera
```

Then manually reposition the arm to 16+ varied poses while keeping one stationary target visible. Press `Enter` to record each pose, then type `save` + `Enter` to fit and write the result to `config/predator_servo_poses.yaml`.

---

## Face Memory

The system automatically enrolls every new person it sees and stores their SFace face embedding in `predator_face_memory.json` at the repo root. This file:

- Is **created and updated at runtime** — you don't create it manually
- **Contains real face embeddings** (biometric data) — see `.gitignore`
- Persists across sessions — the arm remembers who is TARGET and who is BOSS after a restart
- See `predator_face_memory.example.json` for the schema

> **If your repo is public:** `predator_face_memory.json` is gitignored by default to prevent accidentally pushing real face embeddings to GitHub. Keep biometric data out of Git regardless of repo visibility.

---

## What's Gitignored and Why

| Ignored | Reason |
|---------|--------|
| `predator_face_memory.json` | Contains real SFace biometric embeddings — not safe for public repos |
| `face_memory.npz` | Legacy face memory format, superseded by JSON |
| `models/**` | Binary weights are too large for Git (SFace alone is 37 MB). Download separately per the table above |
| `build/`, `install/`, `log/` | ROS 2 colcon build outputs — generated, not source |
| `__pycache__/`, `*.pyc` | Python bytecode cache |

---

## Project Structure

```
ur7e-Predator-Tracking-System/
│
├── src/
│   ├── vision/                      # Vision pipeline (runs on camera machine)
│   │   ├── target_tracker.py        # Main loop: detection, identity, gesture, UDP output
│   │   ├── boss_gesture.py          # Open-palm gesture recognition (MediaPipe)
│   │   ├── vision_runtime.py        # Camera threading, rate limiter, face cache helpers
│   │   ├── shoulder_framing.py      # Background shoulder width estimation (MediaPipe pose)
│   │   └── body_framing.py          # Bounding box height fraction + clip quality
│   │
│   └── control/                     # Arm controller (runs on ROS 2 machine)
│       ├── vision_moveit_servo.py   # ROS 2 node: UDP → IK → MoveIt Servo → UR driver
│       ├── planar_tracking.py       # Analytic 4-joint planar IK + incremental tracker
│       ├── whole_arm_tracking.py    # URDF FK, visual Jacobian, QP solver, camera calibration
│       ├── reach_control.py         # Forward/back reach from body bounding box size
│       └── target_search.py         # Autonomous pan sweep when target is lost
│
├── launch/
│   └── predator_servo.launch.py     # ROS 2 launch: MoveIt + Servo for UR7e
│
├── config/
│   └── predator_servo_poses.yaml    # Start pose, bow angles, IK/tracking/reach config
│
├── models/                          # Downloaded separately — not in Git
│   ├── yolo/yolo26n.pt
│   ├── face/face_detection_yunet_2023mar.onnx
│   │        face_recognition_sface_2021dec.onnx
│   └── hand/hand_landmarker.task
│            pose_landmarker_lite.task
│
├── tools/
│   └── detect.py                    # Standalone detection demo (no arm, no ROS)
│
├── docs/
│   └── setup/MOVEIT_SERVO_SETUP.md  # Detailed arm bridge setup and tuning guide
│
└── predator_face_memory.example.json  # Schema reference for the face identity database
```

---

## Future Upgrades

- **Faster and more dynamic tracking** — current joint speed caps are deliberately conservative for initial hardware testing. The controller architecture already supports higher gains; future work includes tuning for snappier, more fluid tracking responses and reducing latency between vision input and arm motion

- **Jaw / expression movements via soft gripper** — attach a soft gripper to the tool flange and drive it as an expressive "mouth". The gripper opens and closes in sync with detected events: opens wide when the arm acquires a target, closes slowly when the target is lost, snaps open during the boss palm trigger. Gives the arm a predator-like jaw response without any rigid mechanisms

- **Richer gesture vocabulary** — beyond the current open-palm bow, expand to a set of recognized hand shapes that each trigger a distinct arm behavior (point, recoil, aggressive lean-in). The gesture pipeline already runs MediaPipe landmarks at 10 Hz so adding new gesture classifiers is additive

- **Multi-target priority system** — currently one TARGET at a time. Future version ranks visible people by proximity, motion, and role and switches focus dynamically without manual selection

- **Onboard inference** — move YOLO and face recognition onto the arm's compute (Jetson Orin) to cut the UDP round-trip and run the full pipeline at higher frame rates with lower latency

---

## License

MIT
