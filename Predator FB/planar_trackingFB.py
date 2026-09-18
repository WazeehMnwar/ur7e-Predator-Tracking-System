"""URDF-derived planar IK and four-joint image tracking. No ROS calls.

Pitch here is the lift+elbow+wrist1 coordinate, not an inferred optical pitch.
Radius/height locate the configured URDF link origin (tool0 by default).
"""
from dataclasses import dataclass, field
import math
import numpy as np
from whole_arm_tracking import vector
from reach_control import ReachConfig


@dataclass
class IKConfig:
    locked_wrists: object = field(default_factory=lambda: np.radians([90., -90.]).tolist())
    pan_sign: object = None
    tilt_sign: object = None
    pan_span: float = .7
    pitch_span: float = .5
    gain: float = .12
    pose_gain: float = 1.
    feedforward_gain: float = 1.
    radius: object = None
    height: object = None
    auto_start: bool = True

    def __post_init__(self):
        self.locked_wrists = vector(self.locked_wrists, 2, 'locked_wrists')
        for key in ('pan_sign', 'tilt_sign'):
            if getattr(self, key) not in (None, -1, 1):
                raise ValueError(f'{key} must be -1 or +1, or null before direction verification')
        for key, low, high in [('pan_span', .01, math.pi), ('pitch_span', .01, 1.),
                               ('gain', .01, .3), ('pose_gain', .1, 2.),
                               ('feedforward_gain', 0., 1.)]:
            val = float(getattr(self, key))
            if not math.isfinite(val) or not low <= val <= high:
                raise ValueError(f'{key} must be in [{low}, {high}]')
            setattr(self, key, val)
        for key in ('radius', 'height'):
            val = getattr(self, key)
            if val is not None:
                val = float(val)
                if not math.isfinite(val) or abs(val) > 2 or (key == 'radius' and val <= 0):
                    raise ValueError(f'invalid {key} in meters')
                setattr(self, key, val)
        if not isinstance(self.auto_start, bool):
            raise ValueError('auto_start must be true or false')

    @property
    def configured(self):
        return self.pan_sign is not None and self.tilt_sign is not None


class PlanarIK:
    def __init__(self, model, wrists):
        self.model = model
        self.wrists = vector(wrists, 2, 'wrists')
        q = np.r_[np.zeros(4), self.wrists]
        _, axes, _ = model.link_kinematics(q)
        axis = axes[1]
        if (np.max(np.abs(axes[1:4] - axis)) > 1e-5 or abs(axis[2]) > 1e-5
                or np.linalg.norm(np.cross(axes[0], [0, 0, 1])) > 1e-5):
            raise ValueError('URDF axes are not planar; analytic IK cannot represent this calibrated model')
        self.u = np.cross(axis, [0., 0., 1.])
        self.u /= np.linalg.norm(self.u)
        self.v = np.cross(axis, self.u)
        self.axis = axis
        # Recover rigid offsets from exact URDF FK. This includes locked wrists
        # and fixed tool transforms, rather than assuming a single wrist length.
        rng = np.random.default_rng(713)
        samples = rng.uniform(-2., 2., (32, 3))
        design, points = [], []
        for angles in samples:
            phases = np.cumsum(angles)
            design.append([1., *np.column_stack((np.cos(phases), np.sin(phases))).ravel()])
            points.append(model.link_kinematics(np.r_[0., angles, self.wrists])[0][:3, 3])
        coeff = np.linalg.lstsq(design, points, rcond=None)[0]
        if np.max(np.abs(np.asarray(design) @ coeff - points)) > 1e-5:
            raise ValueError('URDF does not fit planar IK within 10 micrometers')
        self.origin = coeff[0]
        self.a, self.b, self.c = [complex(row @ self.u, row @ self.v) for row in coeff[[1, 3, 5]]]
        for i in (1, 3, 5):
            if np.linalg.norm(coeff[i + 1] - np.cross(axis, coeff[i])) > 1e-5:
                raise ValueError('URDF offsets are incompatible with planar IK')
        self.l1, self.l2 = abs(self.a), abs(self.b)
        if min(self.l1, self.l2) < .01:
            raise ValueError('Invalid planar link lengths')

    def branch(self, q):
        relative = q[2] + np.angle(self.b) - np.angle(self.a)
        if abs(math.sin(relative)) < .05:
            raise ValueError('Initial elbow too close to a planar singularity')
        return 1 if math.sin(relative) > 0 else -1

    def solve(self, pan, radial, height, pitch, reference, branch):
        if not np.all(np.isfinite([pan, radial, height, pitch])):
            raise ValueError('Non-finite IK target')
        # radial is signed projection onto the arm plane's horizontal axis.
        point = radial * self.u + height * np.array([0., 0., 1.])
        relative = point - self.origin
        z = complex(relative @ self.u, relative @ self.v) - self.c * np.exp(1j * pitch)
        cosine = (abs(z)**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        if abs(cosine) > 1. or 1. - cosine*cosine < .05**2:
            raise ValueError('IK target unreachable or too close to elbow singularity')
        elbow = branch * math.acos(float(np.clip(cosine, -1., 1.)))
        shoulder = np.angle(z) - math.atan2(self.l2 * math.sin(elbow), self.l1 + self.l2 * math.cos(elbow))
        lift = shoulder - np.angle(self.a)
        elbow -= np.angle(self.b) - np.angle(self.a)
        lift += 2 * math.pi * round((reference[1] - lift) / (2 * math.pi))
        elbow += 2 * math.pi * round((reference[2] - elbow) / (2 * math.pi))
        wrist1 = pitch - lift - elbow
        result = np.r_[pan, lift, elbow, wrist1, self.wrists]
        if np.max(np.abs(result[1:4] - reference[1:4])) > math.pi:
            raise ValueError('IK discontinuity rejected')
        return result


class IKTracker:
    def __init__(self, config, start, reach=None):
        self.config = config
        self.reach = reach or ReachConfig()
        self.start = vector(start, 6, 'start')
        self.reset()

    def reset(self):
        self.target = None
        self.blocked = False
        self.reach_rate = 0.
        self.reach_status = 'waiting'

    def enter(self, positions, model, limits):
        q = vector(positions, 6, 'positions')
        c = self.config
        self.ik = PlanarIK(model, c.locked_wrists)
        self.elbow_branch = self.ik.branch(self.start)
        if self.ik.branch(q) != self.elbow_branch or np.max(np.abs(q[4:] - c.locked_wrists)) > .04:
            raise ValueError('Tracking entry differs from START elbow branch or locked wrists; run START')
        self.low = model.lower + limits.joint_margin
        self.high = model.upper - limits.joint_margin
        self.low[0] = max(self.low[0], self.start[0] - c.pan_span)
        self.high[0] = min(self.high[0], self.start[0] + c.pan_span)
        self.pitch_limits = np.sum(self.start[1:4]) + np.array([-c.pitch_span, c.pitch_span])
        zero_pan = q.copy(); zero_pan[0] = 0.
        point = model.link_kinematics(zero_pan)[0][:3, 3]
        self.radial = float(point @ self.ik.u)
        if c.radius is not None:
            lateral = float(self.ik.origin @ self.ik.axis)
            if c.radius <= abs(lateral):
                raise ValueError('Requested radius is smaller than lateral tool offset')
            self.radial = math.copysign(math.sqrt(c.radius**2 - lateral**2), self.radial)
        start_point = model.link_kinematics(self.start)[0][:3, 3]
        self.height = float(start_point[2]) if c.height is None else c.height
        if not self.reach.height_min <= self.height <= self.reach.height_max:
            raise ValueError("Fixed IK height outside reach height bounds")
        self.pan = float(q[0]); self.pitch = float(np.sum(q[1:4]))
        self.target = self.ik.solve(self.pan, self.radial, self.height, self.pitch, q, self.elbow_branch)
        if np.any(self.target < self.low) or np.any(self.target > self.high):
            raise ValueError('IK entry target exceeds configured joint limits')

    def velocity(self, error, positions, previous, model, limits, dt):
        c = self.config
        q = vector(positions, 6, 'positions')
        error = vector(error, 2, 'error')
        previous = vector(previous, 6, 'velocity')
        if not c.configured:
            raise ValueError('Set ik.pan_sign and ik.tilt_sign after checking image direction')
        if not math.isfinite(dt) or dt <= 0 or np.any(np.abs(error) > 1):
            raise ValueError('Invalid tracking input')
        dt = min(dt, .04)
        if self.target is None:
            self.enter(q, model, limits)
        self.blocked = False
        old_target = self.target.copy()
        correction = np.sign(error) * np.maximum(np.abs(error) - limits.deadband, 0.)
        self.reach_status = 'holding'
        if np.max(np.abs(self.target - q)) < .04:
            pan_step, tilt_step = c.gain * correction * [c.pan_sign, c.tilt_sign] * dt
            # Backtrack infeasible or excessively fast increments, never change
            # elbow branch or clamp individual IK joints into a different pose.
            accepted = False
            lateral = float(self.ik.origin @ self.ik.axis)
            radius = math.hypot(self.radial, lateral)
            # If START is outside the interval, allow only recovery toward it.
            # Do not project straight onto the bound: that would create a jump.
            radius_step = np.clip(self.reach_rate, -self.reach.max_speed, self.reach.max_speed) * dt
            requested_radius = radius + radius_step
            if radius < self.reach.radius_min:
                requested_radius = max(radius, requested_radius)
            elif radius > self.reach.radius_max:
                requested_radius = min(radius, requested_radius)
            else:
                requested_radius = np.clip(requested_radius, self.reach.radius_min, self.reach.radius_max)
            if radius_step != 0 and abs(requested_radius-radius) < 1e-12:
                self.reach_status = 'radius limit: requested direction blocked'
            # Pointing has priority: retry with zero reach if combined IK fails.
            for reach_step in (requested_radius - radius, 0.):
                for scale in (1., .5, .25, .125, .0625, .03125):
                    pan = float(np.clip(self.pan + scale * pan_step, self.low[0], self.high[0]))
                    pitch = float(np.clip(self.pitch + scale * tilt_step, *self.pitch_limits))
                    r = radius + scale * reach_step
                    if r <= abs(lateral):
                        continue
                    radial = math.copysign(math.sqrt(r*r-lateral*lateral), self.radial)
                    try:
                        target = self.ik.solve(pan, radial, self.height, pitch, self.target, self.elbow_branch)
                    except ValueError:
                        continue
                    if (np.any(target < self.low) or np.any(target > self.high)
                            or np.max(np.abs(target-self.target)) > limits.max_joint_speed * dt):
                        continue
                    self.pan, self.pitch, self.radial, self.target = pan, pitch, radial, target
                    if abs(r-radius) > 1e-12:
                        self.reach_status = 'moving'
                    elif reach_step == 0 and abs(requested_radius-radius) > 1e-12:
                        self.reach_status = 'IK blocked reach; pointing only'
                    accepted = True
                    break
                if accepted:
                    break
            self.blocked = not accepted
            if not accepted: self.reach_status = 'IK/joint limit'
        else:
            self.reach_status = 'waiting for arm to catch up'
        # Follow the IK trajectory's velocity directly, then correct any lag.
        # Previously only lag produced velocity: the arm had to fall behind
        # before moving, then hit the .04-rad catch-up gate repeatedly.
        feedforward = (self.target - old_target) / dt
        desired = c.feedforward_gain * feedforward + c.pose_gain * (self.target - q)
        speed = np.minimum(limits.max_joint_speed, model.speed)
        lower = np.maximum(-speed, np.minimum(0., (self.low-q) / .5))
        upper = np.minimum(speed, np.maximum(0., (self.high-q) / .5))
        previous = np.clip(previous, lower, upper)
        step = limits.max_joint_acceleration * dt
        return np.clip(desired, np.maximum(lower, previous-step), np.minimum(upper, previous+step))
