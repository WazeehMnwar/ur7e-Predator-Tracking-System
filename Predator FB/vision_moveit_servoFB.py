#!/usr/bin/env python3
"""
Predator UR7e controller -- ROS 2 Humble + MoveIt Servo.

Image error -> four-joint planar IK -> bounded JointJog -> MoveIt Servo.
Wrist 2/3 hold configured angles. Motion-enabled launch runs START first
when ik.auto_start is true; tracking waits for startup completion.

UDP packets expected from vision:
    TRACK,error_x,error_y[,age_ms]
    BOX,body_height_fraction,track_id,valid,age_ms
    SHOULDERS,width_fraction,track_id,valid,age_ms
    REACH_OFF
    BOW[,age_ms]
    START[,age_ms]
    STOP[,age_ms]
    TARGET_LOST[,age_ms]
    TARGET_MISSING[,age_ms]  # selected person absent; permits bounded search
"""

import argparse
from collections import deque
import math
from pathlib import Path
import select
import socket
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Int8, Float64MultiArray
from std_srvs.srv import Trigger, SetBool
import yaml

from whole_arm_tracking import ArmModel, TrackingConfig, fit_camera_mount
from planar_trackingFB import IKConfig, IKTracker
from reach_control import ReachConfig, ReachControl
from target_search import TargetSearch, SearchConfig, SEARCH_STATES

try:
    from moveit_msgs.srv import ServoCommandType
except ImportError:
    ServoCommandType = None

try:
    from moveit_msgs.msg import ServoStatus
except ImportError:
    ServoStatus = None

# ---------------------------------------------------------------------------
# CONFIG -- deliberately slow for the first physical test
# ---------------------------------------------------------------------------
UDP_BIND = "127.0.0.1"
UDP_PORT = 5005
POSE_FILE = Path(__file__).with_name("predator_servo_posesFB.yaml")

TRACK_TIMEOUT = 0.35           # total age of the camera observation, seconds

POSE_GAIN = 0.90
MAX_JOINT_RATE = 0.120         # rad/s for START/BOW/RETURN
MAX_JOINT_ACCEL = 0.20         # rad/s^2
POSE_TOLERANCE = 0.025         # rad
POSE_TIMEOUT = 60.0            # seconds
BOW_HOLD = 1.0                 # seconds
BOW_COOLDOWN = 5.0             # seconds
BRAKE_TIME = 0.30              # seconds before bow motion

JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
BOW_INDICES = (1, 2, 3)
ZERO6 = (0.0,) * 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_packet(data):
    try:
        fields = [x.strip() for x in data.decode("ascii").split(",")]
        name = fields[0].upper()
        if name in ("REST", "HOME"):
            name = "START"

        if name in ("BOX", "SHOULDERS") and len(fields) == 5:
            height, identity, valid, age = map(float, fields[1:])
            if (0 <= height <= 1 and math.isfinite(identity) and identity.is_integer()
                    and valid in (0, 1, 2) and 0 <= age <= 400):
                return name, (height, int(identity), int(valid)), age / 1000.
        if name == "REACH_OFF" and len(fields) == 1:
            return name, (), 0.
        if name in ("TRACK", "BOSS_PRESENT") and len(fields) in (3, 4):
            ex = float(fields[1])
            ey = float(fields[2])
            age = float(fields[3]) if len(fields) == 4 else 0.0
            if abs(ex) <= 1.0 and abs(ey) <= 1.0 and 0.0 <= age <= 400.0:
                return name, (ex, ey), age / 1000.

        if name in ("BOW", "START", "STOP", "TARGET_LOST", "TARGET_MISSING") and len(fields) in (1, 2):
            age = float(fields[1]) if len(fields) == 2 else 0.0
            if 0.0 <= age <= 400.0:
                return name, (), age / 1000.
    except (UnicodeError, ValueError, IndexError):
        pass
    return None


def load_poses(path):
    with open(path, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f)
    if not isinstance(root, dict):
        raise ValueError("Pose YAML must contain a mapping")

    poses = root.get("poses", root)
    start = tuple(float(x) for x in poses["start"])
    bow = {name: float(value) for name, value in poses["bow"].items()}

    if len(start) != 6 or not all(math.isfinite(x) and abs(x) <= 2 * math.pi for x in start):
        raise ValueError("START pose must contain six finite joint angles in radians")

    expected = {"shoulder_lift_joint", "elbow_joint", "wrist_1_joint"}
    if set(bow) != expected:
        raise ValueError("BOW must contain only shoulder_lift, elbow, wrist_1")
    if not all(math.isfinite(x) and abs(x) <= 2 * math.pi for x in bow.values()):
        raise ValueError("BOW angles must be finite radians")

    return start, bow


def pose_velocity(target, current, indices):
    velocity = [0.0] * 6
    arrived = True

    for i in indices:
        error = target[i] - current[i]
        if abs(error) > POSE_TOLERANCE:
            arrived = False
        velocity[i] = max(-MAX_JOINT_RATE, min(MAX_JOINT_RATE, POSE_GAIN * error))

    return tuple(velocity), arrived


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class PredatorArm(TargetSearch, Node):
    def __init__(self, enable_motion, command_frame=None, poses_file=POSE_FILE,
                 calibrate_camera=False, enable_reach=False):
        super().__init__("vision_moveit_servo")

        self.enable_motion = enable_motion
        self.calibrate_camera = calibrate_camera
        self.poses_file = Path(poses_file)
        self.start_pose, self.bow_pose = load_poses(self.poses_file)
        settings = yaml.safe_load(self.poses_file.read_text())
        options = dict(settings.get("tracking", {}))
        if command_frame is not None:
            options["camera_link"] = command_frame
        self.tracking_config = TrackingConfig(**options)
        self.ik_config = IKConfig(**settings.get("ik", {}))
        if not np.allclose(self.start_pose[4:], self.ik_config.locked_wrists, atol=1e-6):
            raise ValueError("START wrist angles must match ik.locked_wrists")
        self.search_init(SearchConfig(**settings.get("search", {})))
        self.reach_config = ReachConfig(**settings.get("reach", {}))
        self.reach_control = ReachControl(self.reach_config)
        self.reach_enabled = enable_reach
        self.ik_tracker = IKTracker(self.ik_config, self.start_pose, self.reach_config)
        self.startup_pending = bool(enable_motion and not calibrate_camera and self.ik_config.auto_start)
        self.model = None
        self.model_xml = None
        self.reference_pose = None
        self.last_tick = time.monotonic()
        self.last_report = 0.
        self.last_wait_report = 0.
        self.track_stale = False
        self.servo_status = None
        self.servo_output = ZERO6
        self.servo_output_at = 0.
        self.output_fault = False
        self.recent_calibration = deque(maxlen=60)
        self.calibration_samples = []

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((UDP_BIND, UDP_PORT))

        # Motion-enabled launch runs START after Servo/model/joints are ready.
        self.state = "STOP"
        self.start_complete = not self.startup_pending
        self.phase_at = None

        self.track = None
        self.track_at = 0.0
        self.positions = None
        self.joints_at = 0.0

        self.return_pose = None
        self.bow_target = None
        self.last_bow_at = -1e9

        self.last_jog = ZERO6

        self.servo_ready = False
        self.start_future = None
        self.start_stage = None

        self.twist_pub = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 1
        )
        self.jog_pub = self.create_publisher(
            JointJog, "/servo_node/delta_joint_cmds", 1
        )
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.on_joints, qos_profile_sensor_data
        )
        self.start_client = self.create_client(Trigger, "/servo_node/start_servo")
        if ServoCommandType is not None:
            self.switch_client = self.create_client(ServoCommandType, "/servo_node/switch_command_type")
            self.pause_client = self.create_client(SetBool, "/servo_node/pause_servo")
        self.description_sub = self.create_subscription(
            String, "/robot_description", self.on_description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.status_sub = self.create_subscription(
            ServoStatus if ServoStatus is not None else Int8,
            "/servo_node/status", self.on_servo_status, 10,
        )
        self.output_sub = self.create_subscription(
            Float64MultiArray, "/forward_velocity_controller/commands", self.on_servo_output, 1,
        )

        self.create_timer(0.25, self.try_start_servo)
        self.create_timer(0.02, self.tick)  # 50 Hz

        self.get_logger().info(
            f"STOPPED | motion={'ON' if enable_motion else 'OFF'} | "
            f"FB tracking | reach={self.reach_enabled} | joint cap={self.tracking_config.max_joint_speed:.3f} rad/s"
        )
        if not enable_reach:
            self.get_logger().warn(
                "FORWARD/BACK DISABLED: add --enable-reach to enable body-size framing")
        if calibrate_camera:
            self.get_logger().info(
                "CAMERA SETUP: this process sends NO motion. Keep one target still. "
                "Manually reposition the arm, stop, then press Enter to record. "
                "Collect 16+ varied poses; type save + Enter to fit the mount."
            )
        elif not self.ik_tracker.config.configured:
            self.get_logger().warn("IK tracking needs image direction verification: set ik.pan_sign and ik.tilt_sign")

    # ---------------- ROS feedback ----------------
    def on_joints(self, msg):
        table = dict(zip(msg.name, msg.position))
        if all(name in table and math.isfinite(table[name]) for name in JOINTS):
            self.positions = tuple(float(table[name]) for name in JOINTS)
            self.joints_at = time.monotonic()

    def on_description(self, msg):
        if msg.data == self.model_xml:
            return
        try:
            model = ArmModel(msg.data, self.tracking_config.camera_link)
        except (ValueError, TypeError, KeyError, ET.ParseError) as exc:
            self.get_logger().error(f"Robot model unavailable: {exc}")
            self.model = None
            return
        self.model, self.model_xml = model, msg.data
        self.ik_tracker.reset()
        self.get_logger().info("Loaded all six joints from the running robot description")
        for label, degrees in (("rear", [-70, -40, -49, -96]),
                               ("forward", [-70, -134, 9, -52])):
            pose = np.r_[np.radians(degrees), self.ik_config.locked_wrists]
            point = model.link_kinematics(pose)[0][:3, 3]
            self.get_logger().info(
                f"Screenshot {label}: tool radius={np.linalg.norm(point[:2]):.6f} m, "
                f"height={point[2]:.6f} m (configured locked wrists assumed)")

    def on_servo_status(self, msg):
        code = msg.code if hasattr(msg, "code") else msg.data
        halt_codes = {-1, 2, 5, 6} if hasattr(msg, "code") else {-1, 2, 4, 5}
        self.servo_halted = code in halt_codes
        if self.servo_halted:
            self.manual_stop = True
            self.stop()
            self.halt()
        if code != self.servo_status:
            self.servo_status = code
            # Humble and newer Servo releases use different enum numbering.
            humble = {0: "OK", 1: "slowing near singularity", 2: "HALT: singularity",
                      3: "slowing for collision", 4: "HALT: collision", 5: "HALT: joint limit",
                      6: "leaving singularity"}
            description = msg.message if hasattr(msg, "message") else humble.get(code, f"code {code}")
            self.get_logger().info(f"MoveIt Servo: {description}")

    def on_servo_output(self, msg):
        valid = len(msg.data) == 6 and all(math.isfinite(x) for x in msg.data)
        if valid:
            self.servo_output = tuple(msg.data)
            self.servo_output_at = time.monotonic()
        if self.enable_motion and (not valid or max(abs(x) for x in msg.data)
                                   > self.tracking_config.max_joint_speed + 0.005):
            if not self.output_fault:
                self.get_logger().error(
                    "Servo output exceeds configured joint limit. Stopping; use the updated "
                    "predator_servo.launch.py and matching pose YAML, then restart the bridge."
                )
            self.output_fault = True
            self.stop()
            self.halt()

    def try_start_servo(self):
        if not self.enable_motion or self.servo_ready:
            return
        if self.positions is None or time.monotonic() - self.joints_at > 0.5:
            return

        if self.start_future is not None:
            if not self.start_future.done():
                return
            try:
                result = self.start_future.result()
                if result and result.success:
                    if self.start_stage == "select_joint":
                        self.start_stage = "unpause"
                    else:
                        self.servo_ready = True
                        self.get_logger().info("MoveIt Servo ready for joint commands")
                else:
                    self.get_logger().error("MoveIt Servo refused start request")
                    self.start_stage = None
            except Exception as exc:
                self.get_logger().error(f"MoveIt Servo startup: {exc}")
                self.start_stage = None
            finally:
                self.start_future = None
            return

        if self.start_stage == "unpause":
            if self.pause_client.service_is_ready():
                request = SetBool.Request()
                request.data = False
                self.start_future = self.pause_client.call_async(request)
        elif ServoCommandType is not None and self.switch_client.service_is_ready():
            request = ServoCommandType.Request()
            request.command_type = ServoCommandType.Request.JOINT_JOG
            self.start_stage = "select_joint"
            self.start_future = self.switch_client.call_async(request)
        elif self.start_client.service_is_ready():
            self.start_stage = "start"
            self.start_future = self.start_client.call_async(Trigger.Request())

    # ---------------- Output ----------------
    def publish_twist(self):
        if not self.enable_motion:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        # Only clear old Cartesian input. All actual motion is JointJog.
        self.twist_pub.publish(msg)

    def publish_jog(self, velocity=ZERO6, indices=range(6)):
        if not self.enable_motion:
            return
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.joint_names = [JOINTS[i] for i in indices]
        msg.velocities = [float(velocity[i]) for i in indices]
        self.jog_pub.publish(msg)

    def halt(self):
        self.last_jog = ZERO6
        self.publish_twist()
        self.publish_jog(ZERO6)

    # ---------------- State ----------------
    def set_state(self, new_state):
        if hasattr(self, "ik_tracker"):
            self.ik_tracker.reset()
            self.reach_control.reset()
        if new_state != self.state:
            self.get_logger().info(f"{self.state} -> {new_state}")
        self.state = new_state
        self.phase_at = None
        self.last_jog = ZERO6

    def stop(self):
        self.cancel_search()
        self.startup_pending = False
        self.track = None
        self.set_state("STOP")

    # ---------------- UDP ----------------
    def read_udp(self, now):
        for _ in range(64):
            try:
                packet, _ = self.sock.recvfrom(256)
            except BlockingIOError:
                break

            parsed = parse_packet(packet)
            if parsed is None:
                continue
            name, values, age = parsed

            if name in ("BOX", "SHOULDERS"):
                expected = 'SHOULDERS' if self.reach_control.config.metric == 'shoulder_width' else 'BOX'
                if name == expected:
                    self.reach_control.observe(*values, now - age)
                continue
            if name == "REACH_OFF":
                self.reach_enabled = False
                self.reach_control.reset()
                self.ik_tracker.reach_rate = 0.
                self.get_logger().info("Reach disabled; pointing remains active")
                continue
            if name == "TRACK":
                self.record_track(values, now-age)
                self.track = values
                self.track_at = now - age
                if (self.calibrate_camera and self.positions is not None
                        and now - self.joints_at < 0.1 and age < 0.15):
                    self.recent_calibration.append((now, self.positions, values))
                if (
                    self.enable_motion
                    and not self.output_fault
                    and not self.manual_stop and not self.servo_halted
                    and self.servo_ready
                    and self.start_complete
                    and self.state in ({"STOP"} | (SEARCH_STATES - {"SEARCH_HOME"}))
                ):
                    self.set_state("TRACK")

            elif name == "BOW":
                self.begin_bow(now)

            elif name == "START":
                if (self.enable_motion and self.servo_ready and not self.output_fault
                        and not self.servo_halted
                        and self.state in ({"STOP", "TRACK"} | SEARCH_STATES)):
                    self.cancel_search()
                    self.manual_stop = False
                    self.startup_pending = False
                    self.start_complete = False
                    self.set_state("START")
                else:
                    self.get_logger().warn(
                        f"START rejected: motion={self.enable_motion}, Servo ready={self.servo_ready}, "
                        f"fault={self.output_fault}, state={self.state}; resend after ready")

            elif name == "STOP":
                self.manual_stop = True
                self.stop()
                self.halt()
            elif name == "TARGET_MISSING":
                self.track = None
                self.missing_target(now, now-age)
            elif name == "TARGET_LOST":
                self.cancel_search()
                self.track = None
                self.recent_calibration.clear()
                if self.state == "TRACK" or self.state in SEARCH_STATES:
                    self.stop()
                    self.halt()

            # BOSS_PRESENT deliberately does nothing here.
            # Vision decides when a verified gesture becomes BOW.

    # ---------------- Tracking ----------------
    def run_track(self, now, dt=0.02):
        if self.track is None or now - self.track_at > TRACK_TIMEOUT:
            self.halt()
            # Keep the bounded IK target during a short gap. Zero output still
            # takes effect immediately; fresh vision need not rebuild the model
            # or restart the trajectory from zero on every brief frame delay.
            self.ik_tracker.reach_rate = 0.
            self.reach_control.pause('stale detection', reacquire=True)
            if not self.track_stale:
                self.get_logger().warn("Tracking paused: image older than 0.35s; holding for fresh vision")
                self.track_stale = True
            if self.track is None or now-self.track_at > self.search_config.heartbeat_timeout:
                # No motion during an outage. Preserve one search opportunity
                # only until vision explicitly confirms that the target is still
                # selected but absent. STOP/clearing/faults still disarm it.
                armed = self.search_armed
                history = tuple(self.target_history)
                self.stop()
                self.search_armed = armed
                self.target_history.extend(history)
                self.get_logger().warn("Tracking stopped: vision outage; no search without TARGET_MISSING heartbeat")
            return
        if self.track_stale:
            self.get_logger().info("Fresh vision restored; tracking resumed")
            self.track_stale = False

        if self.model is None or not self.ik_tracker.config.configured:
            self.halt()
            if now - self.last_wait_report > 2.:
                reason = ("waiting for /robot_description" if self.model is None else
                          "set ik.pan_sign and ik.tilt_sign (+1 or -1); no posture table needed")
                self.get_logger().warn(reason)
                self.last_wait_report = now
            return
        if self.reference_pose is None:
            self.reference_pose = tuple(self.positions)
        try:
            if self.ik_tracker.target is None:
                self.ik_tracker.enter(self.positions, self.model, self.tracking_config)
            self.ik_tracker.reach_rate = (self.reach_control.rate(now, self.track, self.track_at)
                                          if self.reach_enabled else 0.)
            requested = self.ik_tracker.velocity(
                self.track, self.positions, self.last_jog,
                self.model, self.tracking_config, dt,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            self.output_fault = True
            self.stop()
            self.halt()
            self.get_logger().error(f"Tracking stopped: {exc}")
            return
        self.last_jog = tuple(float(x) for x in requested)
        self.publish_twist()
        self.publish_jog(self.last_jog)
        if now - self.last_report >= 2.:
            output = (",".join(f"{v:+.3f}" for v in self.servo_output)
                      if now - self.servo_output_at < 0.5 else "waiting for Servo output")
            box = self.reach_control.filtered
            box_text = "none" if box is None else f"{box:.3f}"
            self.get_logger().info(
                f"TRACK error=({self.track[0]:+.3f},{self.track[1]:+.3f}) | "
                "Servo joints [pan,lift,elbow,w1,w2,w3] rad/s="
                + output + f" | {self.reach_config.metric}={box_text} reach={self.reach_enabled} requested={self.ik_tracker.reach_rate:+.4f} m/s"
                + f" ({self.reach_control.diagnostic(now) if self.reach_enabled else 'disabled; add --enable-reach'})"
                + f" | framing: {self.reach_control.diagnostic(now)}"
                + f" | radius={math.hypot(self.ik_tracker.radial, self.ik_tracker.ik.origin @ self.ik_tracker.ik.axis):.3f}m {self.ik_tracker.reach_status}"
                + (" | IK limit: holding last feasible target" if self.ik_tracker.blocked else "")
            )
            self.last_report = now

    def calibration_input(self, now):
        if not select.select([sys.stdin], [], [], 0)[0]:
            return
        line = sys.stdin.readline()
        if not line:
            return
        if line.strip().lower() == "save":
            if self.model is None:
                self.get_logger().warn("Waiting for /robot_description")
                return
            self.get_logger().info("Fitting mount and validating held-out samples...")
            try:
                result, rms = fit_camera_mount(self.calibration_samples, self.model, self.tracking_config)
                root = yaml.safe_load(self.poses_file.read_text())
                root.setdefault("tracking", {}).update(result)
                root["tracking"]["camera_link"] = self.tracking_config.camera_link
                # Validate before replacing any configuration on disk.
                TrackingConfig(**root["tracking"])
                temporary = self.poses_file.with_suffix(".yaml.tmp")
                temporary.write_text(yaml.safe_dump(root, sort_keys=False))
                temporary.replace(self.poses_file)
                self.get_logger().info(
                    f"Camera setup saved to {self.poses_file}; validation RMS={rms:.4f}. "
                    "Stop this process and restart with --enable-motion."
                )
            except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
                self.get_logger().warn(f"Setup not saved: {exc}")
            return
        if line.strip():
            self.get_logger().info("Enter records a pose; save + Enter fits the camera mount")
            return
        recent = [s for s in self.recent_calibration if now - s[0] < 0.6]
        if (len(recent) < 3 or recent[-1][0] - recent[0][0] < 0.2
                or now - recent[-1][0] > 0.15):
            self.get_logger().warn("Need a fresh selected target; wait still briefly before Enter")
            return
        q = np.array([s[1] for s in recent])
        image = np.array([s[2] for s in recent])
        if np.max(np.ptp(q, axis=0)) > 0.005 or np.max(np.ptp(image, axis=0)) > 0.025:
            self.get_logger().warn("Robot or target still moving; wait before recording")
            return
        sample = (np.mean(q, axis=0), np.mean(image, axis=0))
        if any(np.linalg.norm(sample[0] - old[0]) < 0.025 for old in self.calibration_samples):
            self.get_logger().warn("Pose already recorded; use a different arm/wrist pose")
            return
        self.calibration_samples.append(sample)
        self.get_logger().info(f"Recorded camera sample {len(self.calibration_samples)} (need 16+)")

    # ---------------- START / BOW / RETURN ----------------
    def begin_bow(self, now):
        if (
            not self.enable_motion
            or self.output_fault
            or self.manual_stop or self.servo_halted
            or not self.servo_ready
            or not self.start_complete
            or self.state not in ({"STOP", "TRACK"} | SEARCH_STATES)
            or self.positions is None
            or now - self.joints_at > 0.3
            or now - self.last_bow_at < BOW_COOLDOWN
        ):
            self.get_logger().warn(
                f"BOW rejected: motion={self.enable_motion}, Servo ready={self.servo_ready}, "
                f"fault={self.output_fault}, start complete={self.start_complete}, state={self.state}; "
                "requires fresh joints and expired cooldown")
            return

        # Save the exact current six-joint pose.
        self.cancel_search()
        self.return_pose = tuple(self.positions)

        # Bow changes ONLY shoulder_lift, elbow, wrist_1.
        target = list(self.return_pose)
        for joint, angle in self.bow_pose.items():
            target[JOINTS.index(joint)] = angle
        self.bow_target = tuple(target)

        self.last_bow_at = now
        self.set_state("BRAKE")
        self.phase_at = now
        self.get_logger().info("BOW accepted: saved six-joint pose")

    def run_pose(self, now, target, indices):
        if self.phase_at is None:
            self.phase_at = now
        if now - self.phase_at > POSE_TIMEOUT:
            self.get_logger().error(f"{self.state} timeout -> STOP")
            self.stop()
            self.halt()
            return

        requested, arrived = pose_velocity(target, self.positions, indices)
        if arrived:
            self.halt()
            if self.state in ("SEARCH_HOME", "SEARCH_PAN"):
                self.search_pose_arrived(now)
            elif self.state == "START":
                self.start_complete = True
                self.reference_pose = tuple(self.positions)
                self.set_state("STOP")
            elif self.state == "LOWER":
                self.set_state("HOLD")
                self.phase_at = now
            elif self.state == "RETURN":
                self.set_state("STOP")
            return

        # Smooth joint speed for pose behaviors.
        step = min(MAX_JOINT_ACCEL, self.tracking_config.max_joint_acceleration) * 0.02
        smoothed = list(ZERO6)
        for i in indices:
            smoothed[i] = max(
                self.last_jog[i] - step,
                min(self.last_jog[i] + step, requested[i]),
            )
            smoothed[i] = max(-self.tracking_config.max_joint_speed,
                              min(self.tracking_config.max_joint_speed, smoothed[i]))
        self.last_jog = tuple(smoothed)

        # Humble Servo prioritizes a NONZERO Cartesian command, so keep
        # Cartesian zero while JointJog is active.
        self.publish_twist()
        self.publish_jog(self.last_jog, indices)

    # ---------------- Main 50 Hz loop ----------------
    def tick(self):
        now = time.monotonic()
        dt = max(0.001, min(now - self.last_tick, 0.04))
        self.last_tick = now
        self.read_udp(now)

        if self.calibrate_camera:
            self.calibration_input(now)
            return

        if not self.enable_motion or not self.servo_ready:
            return
        if self.output_fault or self.servo_halted or self.manual_stop:
            self.halt()
            return

        if self.positions is None or now - self.joints_at > 0.3:
            if not getattr(self, "startup_pending", False):
                self.stop()
            self.halt()
            return

        if getattr(self, "startup_pending", False):
            if self.model is None:
                self.halt()
                return
            low = self.model.lower + self.tracking_config.joint_margin
            high = self.model.upper - self.tracking_config.joint_margin
            if np.any(np.array(self.start_pose) < low) or np.any(np.array(self.start_pose) > high):
                self.output_fault = True
                self.stop()
                self.halt()
                self.get_logger().error("START exceeds URDF joint limits")
                return
            self.startup_pending = False
            self.set_state("START")
            self.get_logger().info("Automatic START: tracking waits until the start pose is reached")

        if self.state == "STOP":
            self.halt()

        elif self.state == "TRACK":
            self.run_track(now, dt)

        elif self.state in SEARCH_STATES:
            self.run_search(now, dt)

        elif self.state == "START":
            self.run_pose(now, self.start_pose, range(6))

        elif self.state == "BRAKE":
            self.halt()
            if now - self.phase_at >= BRAKE_TIME:
                self.set_state("LOWER")

        elif self.state == "LOWER":
            self.run_pose(now, self.bow_target, BOW_INDICES)

        elif self.state == "HOLD":
            self.halt()
            if now - self.phase_at >= BOW_HOLD:
                self.set_state("RETURN")

        elif self.state == "RETURN":
            self.run_pose(now, self.return_pose, range(6))

    def destroy_node(self):
        if rclpy.ok() and self.enable_motion:
            for _ in range(10):
                self.halt()
                time.sleep(0.02)
        self.sock.close()
        return super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--enable-reach", action="store_true", help="Enable slow body-box framing; off by default")
    parser.add_argument("--command-frame", help="Override the camera's carrying URDF link")
    parser.add_argument("--poses", type=Path, default=POSE_FILE)
    parser.add_argument("--calibrate-camera", action="store_true",
                        help="Record manual poses; never send robot motion")
    args = parser.parse_args()
    if args.calibrate_camera and args.enable_motion:
        parser.error("--calibrate-camera cannot be combined with --enable-motion")

    rclpy.init()
    node = PredatorArm(args.enable_motion, args.command_frame, args.poses, args.calibrate_camera, args.enable_reach)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
