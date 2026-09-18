"""Track the selected person with a UR arm using ROS 2 Humble.

Run separately from target_tracker.py with the UR driver's
forward_velocity_controller active. This script never moves to a preset pose.
"""

import argparse
import math
import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
ZERO = [0.0] * 6


class VisionArmTracker(Node):
    def __init__(self, args):
        super().__init__("vision_arm_tracker")
        self.args = args
        self.positions = None
        self.joint_time = 0.0
        self.track = None
        self.track_time = 0.0
        self.last_tick = time.monotonic()
        self.last_velocity = [0.0] * 6
        self.velocity_pub = self.create_publisher(
            Float64MultiArray, "/forward_velocity_controller/commands", 1
        )
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.on_joints, qos_profile_sensor_data
        )
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((args.bind, args.port))
        self.timer = self.create_timer(0.02, self.tick)
        self.get_logger().info(
            f"Tracking UDP on {args.bind}:{args.port}; "
            f"motion {'enabled' if args.enable_motion else 'disabled'}"
        )

    def on_joints(self, message):
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in JOINTS):
            return
        values = [positions[name] for name in JOINTS]
        if not all(math.isfinite(value) for value in values):
            return
        self.positions = values
        self.joint_time = time.monotonic()

    def read_vision(self, now):
        # Drain the socket so control uses the freshest observation.
        for _ in range(64):
            try:
                packet, address = self.sock.recvfrom(256)
            except BlockingIOError:
                break
            if self.args.sender_ip and address[0] != self.args.sender_ip:
                continue
            try:
                fields = packet.decode("ascii").strip().split(",")
                command = fields[0].strip().upper()
                if command in ("TARGET_LOST", "STOP"):
                    self.track = None
                    continue
                if command != "TRACK" or len(fields) not in (3, 4):
                    continue  # BOSS_PRESENT and BOW never move this controller.
                x, y = float(fields[1]), float(fields[2])
                age_ms = float(fields[3]) if len(fields) == 4 else 0.0
                if (not all(math.isfinite(v) for v in (x, y, age_ms))
                        or abs(x) > 1 or abs(y) > 1
                        or not 0 <= age_ms <= 500):
                    continue
                self.track = (x, y)
                self.track_time = now
            except (UnicodeError, ValueError, IndexError):
                continue

    def tick(self):
        now = time.monotonic()
        self.read_vision(now)
        desired = ZERO.copy()
        fresh = (self.args.enable_motion and self.positions is not None
                 and now - self.joint_time < 0.3 and self.track is not None
                 and now - self.track_time < 0.5)
        if fresh:
            x, y = self.track
            pan, lift = self.positions[:2]
            if abs(x) > self.args.deadband:
                desired[0] = self.args.pan_sign * self.args.pan_gain * x
            if abs(y) > self.args.deadband:
                desired[1] = self.args.lift_sign * self.args.lift_gain * y
            # These are broad joint bounds; set narrower bounds for your cell.
            if (pan <= self.args.pan_min and desired[0] < 0
                    or pan >= self.args.pan_max and desired[0] > 0):
                desired[0] = 0.0
            if (lift <= self.args.lift_min and desired[1] < 0
                    or lift >= self.args.lift_max and desired[1] > 0):
                desired[1] = 0.0

        dt = min(0.1, max(0.0, now - self.last_tick))
        self.last_tick = now
        if fresh:
            for i in (0, 1):
                target = max(-self.args.max_speed,
                             min(self.args.max_speed, desired[i]))
                change = self.args.max_accel * dt
                desired[i] = max(self.last_velocity[i] - change,
                                 min(self.last_velocity[i] + change, target))
            # The position limit also applies after acceleration limiting.
            pan, lift = self.positions[:2]
            if (pan <= self.args.pan_min and desired[0] < 0
                    or pan >= self.args.pan_max and desired[0] > 0):
                desired[0] = 0.0
            if (lift <= self.args.lift_min and desired[1] < 0
                    or lift >= self.args.lift_max and desired[1] > 0):
                desired[1] = 0.0
        # Loss of vision, joint feedback, or motion enable sends zero now.
        self.last_velocity = desired
        self.velocity_pub.publish(Float64MultiArray(data=desired))

    def destroy_node(self):
        if rclpy.ok():
            self.velocity_pub.publish(Float64MultiArray(data=ZERO.copy()))
        self.sock.close()
        return super().destroy_node()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--sender-ip", default="")
    parser.add_argument("--pan-gain", type=float, default=0.08)
    parser.add_argument("--lift-gain", type=float, default=0.06)
    parser.add_argument("--pan-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--lift-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--deadband", type=float, default=0.05)
    parser.add_argument("--max-speed", type=float, default=0.08)
    parser.add_argument("--max-accel", type=float, default=0.2)
    parser.add_argument("--pan-min", type=float, default=-6.0)
    parser.add_argument("--pan-max", type=float, default=6.0)
    parser.add_argument("--lift-min", type=float, default=-6.0)
    parser.add_argument("--lift-max", type=float, default=6.0)
    args = parser.parse_args()
    if (not 1 <= args.port <= 65535
            or args.bind not in ("127.0.0.1", "localhost") and not args.sender_ip
            or not 0 <= args.deadband < 1
            or not 0 < args.max_speed <= 0.5
            or not 0 < args.max_accel <= 2.0
            or not 0 <= args.pan_gain <= 1 or not 0 <= args.lift_gain <= 1
            or not args.pan_min < args.pan_max
            or not args.lift_min < args.lift_max
            or not all(math.isfinite(v) for v in (
                args.deadband, args.max_speed, args.max_accel,
                args.pan_gain, args.lift_gain, args.pan_min, args.pan_max,
                args.lift_min, args.lift_max,
            ))):
        parser.error("invalid tracking parameter or bind address")
    return args


def main():
    args = arguments()
    rclpy.init()
    node = VisionArmTracker(args)
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
