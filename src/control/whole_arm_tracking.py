"""Image-based joint control with a preference for moving the arm's body.

No ROS or camera capture here. The bridge supplies the URDF, live joints and
image error, then sends the bounded result to MoveIt Servo as JointJog.
"""

from dataclasses import dataclass, field
import math
import xml.etree.ElementTree as ET

import numpy as np


JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


@dataclass
class TrackingConfig:
    camera_link: str = "tool0"
    # Directions in camera_link, not the camera's position in the room.
    camera_right: object = None
    camera_down: object = None
    camera_offset: object = field(default_factory=lambda: [0.0, 0.0, 0.0])
    horizontal_fov_deg: float = 60.0
    vertical_fov_deg: float = 45.0
    estimated_depth_m: float = 1.5
    response_gain: float = 0.8
    deadband: float = 0.02
    # Larger cost means use that joint less. All six remain available.
    joint_cost: object = field(default_factory=lambda: [1., 1., 1., 3., 8., 8.])
    wrist_posture_gain: float = 0.15
    max_joint_speed: float = 0.10
    max_joint_acceleration: float = 0.20
    damping: float = 0.08
    joint_margin: float = 0.12

    def __post_init__(self):
        limits = {
            "horizontal_fov_deg": (10., 150.), "vertical_fov_deg": (10., 150.),
            "estimated_depth_m": (0.2, 10.), "response_gain": (0.05, 2.),
            "deadband": (0., 0.2), "wrist_posture_gain": (0., 1.),
            "max_joint_speed": (0.01, 0.20),
            "max_joint_acceleration": (0.01, 0.5), "damping": (0.01, 1.),
            "joint_margin": (0.02, 0.5),
        }
        for name, (low, high) in limits.items():
            val = float(getattr(self, name))
            if not math.isfinite(val) or not low <= val <= high:
                raise ValueError(f"tracking.{name} must be in [{low}, {high}]")
            setattr(self, name, val)
        self.joint_cost = vector(self.joint_cost, 6, "joint_cost")
        if np.any(self.joint_cost < 1) or np.any(self.joint_cost > 50):
            raise ValueError("joint_cost entries must be in [1, 50]")
        self.camera_offset = vector(self.camera_offset, 3, "camera_offset")
        if np.linalg.norm(self.camera_offset) > 1:
            raise ValueError("camera_offset must be in meters, within 1 m of its link")
        if not isinstance(self.camera_link, str) or not self.camera_link:
            raise ValueError("camera_link must name a URDF link")
        if (self.camera_right is None) != (self.camera_down is None):
            raise ValueError("set both camera_right and camera_down")
        if self.camera_right is not None:
            self.camera_right = vector(self.camera_right, 3, "camera_right")
            self.camera_down = vector(self.camera_down, 3, "camera_down")
            if (abs(np.linalg.norm(self.camera_right) - 1.) > 1e-3
                    or abs(np.linalg.norm(self.camera_down) - 1.) > 1e-3
                    or abs(self.camera_right @ self.camera_down) > 1e-3):
                raise ValueError("camera_right/down must be perpendicular unit vectors")

    @property
    def camera_configured(self):
        return self.camera_right is not None and self.camera_down is not None


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * skew + (1. - math.cos(angle)) * (skew @ skew)


class ArmModel:
    """FK and camera Jacobian from the running robot's URDF and joint order."""

    def __init__(self, urdf, camera_link="tool0", base_link="base_link"):
        root = ET.fromstring(urdf)
        parents = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain = []
        current = camera_link
        visited = set()
        while current != base_link:
            if current in visited or current not in parents:
                raise ValueError(f"No URDF chain from {base_link} to {camera_link}")
            visited.add(current)
            joint = parents[current]
            chain.append(joint)
            current = joint.find("parent").get("link")
        self.chain = []
        self.lower = np.full(6, -np.inf)
        self.upper = np.full(6, np.inf)
        self.speed = np.full(6, np.inf)
        active = []
        for joint in reversed(chain):
            kind, name = joint.get("type"), joint.get("name")
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError(f"Unsupported joint type in camera chain: {kind}")
            index = None if kind == "fixed" else JOINTS.index(name)
            origin = joint.find("origin")
            xyz = vector([float(x) for x in (origin.get("xyz", "0 0 0")
                         if origin is not None else "0 0 0").split()], 3, "URDF xyz")
            rpy = vector([float(x) for x in (origin.get("rpy", "0 0 0")
                         if origin is not None else "0 0 0").split()], 3, "URDF rpy")
            transform = np.eye(4)
            transform[:3, :3] = (rotation([0, 0, 1], rpy[2])
                                  @ rotation([0, 1, 0], rpy[1])
                                  @ rotation([1, 0, 0], rpy[0]))
            transform[:3, 3] = xyz
            axis_node = joint.find("axis")
            axis = vector([float(x) for x in (axis_node.get("xyz", "1 0 0")
                          if axis_node is not None else "1 0 0").split()], 3, "URDF axis")
            if np.linalg.norm(axis) < 1e-8:
                raise ValueError(f"Zero axis for {name}")
            axis /= np.linalg.norm(axis)
            if index is not None:
                active.append(name)
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError(f"Missing limits for {name}")
                self.speed[index] = float(limit.get("velocity"))
                if not math.isfinite(self.speed[index]) or self.speed[index] <= 0:
                    raise ValueError(f"Invalid velocity limit for {name}")
                if kind == "revolute":
                    self.lower[index] = float(limit.get("lower"))
                    self.upper[index] = float(limit.get("upper"))
                    if (not math.isfinite(self.lower[index])
                            or not math.isfinite(self.upper[index])
                            or self.lower[index] >= self.upper[index]):
                        raise ValueError(f"Invalid position limits for {name}")
            self.chain.append((index, transform, axis))
        if tuple(active) != JOINTS:
            raise ValueError("Camera chain must include all six UR joints in order")

    def link_kinematics(self, positions):
        q = vector(positions, 6, "joint positions")
        transform = np.eye(4)
        axes, origins = np.zeros((6, 3)), np.zeros((6, 3))
        for index, origin, axis in self.chain:
            transform = transform @ origin
            if index is not None:
                origins[index] = transform[:3, 3]
                axes[index] = transform[:3, :3] @ axis
                turn = np.eye(4)
                turn[:3, :3] = rotation(axis, q[index])
                transform = transform @ turn
        return transform, axes, origins

    def camera_kinematics(self, positions, config):
        if not config.camera_configured:
            raise ValueError("Set camera_right/down in the tracking configuration")
        transform, axes, origins = self.link_kinematics(positions)
        camera = transform.copy()
        camera[:3, 3] += transform[:3, :3] @ config.camera_offset
        mount = np.column_stack((config.camera_right, config.camera_down,
                                 np.cross(config.camera_right, config.camera_down)))
        camera[:3, :3] = transform[:3, :3] @ mount
        jacobian = np.empty((6, 6))
        for i in range(6):
            jacobian[:3, i] = camera[:3, :3].T @ np.cross(axes[i], camera[:3, 3] - origins[i])
            jacobian[3:, i] = camera[:3, :3].T @ axes[i]
        return camera, jacobian


def image_jacobian(feature, depth, camera_jacobian):
    """Point interaction matrix: image feature rate = L * J * joint speed."""
    x, y = feature
    interaction = np.array([
        [-1. / depth, 0., x / depth, x * y, -(1. + x * x), y],
        [0., -1. / depth, y / depth, 1. + y * y, -x * y, -x],
    ])
    return interaction @ camera_jacobian


def bounded_quadratic(hessian, gradient, lower, upper):
    """Small strictly convex box QP, solved with a feasible active set."""
    value = np.clip(np.linalg.solve(hessian, -gradient), lower, upper)
    active = np.zeros(len(value), dtype=int)
    active[value <= lower + 1e-12] = -1
    active[value >= upper - 1e-12] = 1
    fixed = upper - lower < 1e-12
    for _ in range(100):
        free = np.flatnonzero(active == 0)
        bound = np.flatnonzero(active != 0)
        candidate = value.copy()
        if len(free):
            candidate[free] = np.linalg.solve(
                hessian[np.ix_(free, free)],
                -gradient[free] - hessian[np.ix_(free, bound)] @ value[bound],
            )
        delta = candidate - value
        fraction, hit, side = 1., None, 0
        for i in free:
            if candidate[i] < lower[i] - 1e-12:
                step = (lower[i] - value[i]) / delta[i]
                if step < fraction:
                    fraction, hit, side = step, i, -1
            elif candidate[i] > upper[i] + 1e-12:
                step = (upper[i] - value[i]) / delta[i]
                if step < fraction:
                    fraction, hit, side = step, i, 1
        value = np.clip(value + fraction * delta, lower, upper)
        if hit is not None:
            active[hit] = side
            continue
        grad = hessian @ value + gradient
        violation = np.where(active == -1, -grad, np.where(active == 1, grad, 0.))
        violation[fixed] = 0.
        release = int(np.argmax(violation))
        if violation[release] <= 1e-9:
            return value
        active[release] = 0
    raise RuntimeError("Joint tracking optimizer failed to converge")


def tracking_velocity(error, positions, previous, reference, model, config, dt):
    """Keep the image centered, prefer body motion and avoid wrist winding."""
    error = vector(error, 2, "image error")
    q = vector(positions, 6, "joint positions")
    previous = vector(previous, 6, "previous velocity")
    reference = vector(reference, 6, "posture reference")
    if np.any(np.abs(error) > 1) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid image error or control interval")
    dt = min(dt, 0.04)  # A delayed tick must not permit a velocity jump.
    speed = np.minimum(config.max_joint_speed, model.speed)
    lower = np.maximum(-speed, np.minimum(0., (model.lower + config.joint_margin - q) / 0.5))
    upper = np.minimum(speed, np.maximum(0., (model.upper - config.joint_margin - q) / 0.5))
    # Joint-limit stopping takes priority over the acceleration ramp.
    previous = np.clip(previous, lower, upper)
    lower = np.maximum(lower, previous - config.max_joint_acceleration * dt)
    upper = np.minimum(upper, previous + config.max_joint_acceleration * dt)
    if np.all(np.abs(error) <= config.deadband):
        return np.clip(np.zeros(6), lower, upper)
    scales = np.tan(np.radians([config.horizontal_fov_deg, config.vertical_fov_deg]) / 2.)
    feature = error * scales
    correction = np.sign(error) * np.maximum(np.abs(error) - config.deadband, 0.) * scales
    _, camera_j = model.camera_kinematics(q, config)
    visual_j = image_jacobian(feature, config.estimated_depth_m, camera_j)
    # Softly discourage moving toward the person and rolling the image. There
    # is no constraint holding the arm tip at one XYZ location or orientation.
    task = np.vstack((visual_j, 0.15 * camera_j[2], 0.15 * camera_j[5]))
    desired = np.r_[-config.response_gain * correction, 0., 0.]
    posture = np.zeros(6)
    posture[3:] = config.wrist_posture_gain * (reference[3:] - q[3:])
    regularization = config.damping ** 2 * config.joint_cost ** 2
    hessian = task.T @ task + np.diag(regularization)
    gradient = -task.T @ desired - regularization * posture
    result = bounded_quadratic(hessian, gradient, lower, upper)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Non-finite tracking joint command")
    return result


def fit_camera_mount(samples, model, config):
    """Fit a fixed mount from manual arm poses looking at ONE stationary point.

    Each sample is (six joint positions, two normalized image coordinates).
    This never sends motion. Fit rotation, mounting offset and the stationary
    point and approximate field of view together, then check observability and
    samples withheld from fitting. No measured mounting coordinates are needed.
    """
    import itertools

    if len(samples) < 16:
        raise ValueError("Need at least 16 distinct, stationary arm poses")
    joints = np.array([vector(s[0], 6, "sample joints") for s in samples])
    pixels = np.array([vector(s[1], 2, "sample image") for s in samples])
    if np.any(np.abs(pixels) > 1):
        raise ValueError("Calibration image samples must be inside the camera frame")
    spread = np.ptp(joints, axis=0)
    if (np.count_nonzero(spread > 0.08) < 3 or not np.any(spread[:3] > 0.08)
            or not np.any(spread[3:] > 0.08)):
        raise ValueError("Samples must vary at least three joints, including body and wrist motion")
    transforms = np.array([model.link_kinematics(q)[0] for q in joints])
    rotations, positions = transforms[:, :3, :3], transforms[:, :3, 3]
    if np.linalg.norm(np.ptp(positions, axis=0)) < 0.06:
        raise ValueError("Need more arm translation between poses (at least 6 cm spread)")
    scales = np.tan(np.radians([config.horizontal_fov_deg, config.vertical_fov_deg]) / 2.)
    rays = np.column_stack((pixels * scales, np.ones(len(pixels))))
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    training = np.arange(len(samples)) % 4 != 0

    def matrix(rotvec):
        angle = np.linalg.norm(rotvec)
        return np.eye(3) if angle < 1e-12 else rotation(rotvec / angle, angle)

    def predict(parameters, seed):
        mount = seed @ matrix(parameters[:3])
        # R_link.T * (point_world - link_position) - mount_offset.
        local = np.einsum('nji,nj->ni', rotations, parameters[6:9] - positions) - parameters[3:6]
        camera = local @ mount
        lengths = np.linalg.norm(camera, axis=1)
        unit = camera / np.maximum(lengths[:, None], 1e-9)
        return unit, camera, mount

    def residual(parameters, seed):
        unit, _, _ = predict(parameters, seed)
        measured = np.column_stack((pixels * np.exp(parameters[9:11]), np.ones(len(pixels))))
        measured /= np.linalg.norm(measured, axis=1)[:, None]
        scale = np.exp(parameters[9:11])
        # Normalize by focal scale so shrinking the fitted FOV cannot make a
        # bad mount appear accurate merely by shrinking all measured rays.
        weights = np.r_[scale, min(scale)]
        return ((unit[training] - measured[training]) / weights).ravel()

    def derivative(parameters, seed, value):
        result = np.empty((len(value), 11))
        for i in range(11):
            perturbed = parameters.copy()
            perturbed[i] += 1e-5
            result[:, i] = (residual(perturbed, seed) - value) / 1e-5
        return result

    best = None
    # Cover every axis permutation and sign; the mount need not be aligned.
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            seed = np.eye(3)[:, perm] @ np.diag(signs)
            if np.linalg.det(seed) < 0:
                continue
            world_rays = np.einsum('nij,nj->ni', rotations, rays @ seed.T)
            point = np.mean(positions + config.estimated_depth_m * world_rays, axis=0)
            parameters = np.r_[np.zeros(6), point, np.log(scales)]
            damping = 1e-3
            for _ in range(100):
                value = residual(parameters, seed)
                jacobian = derivative(parameters, seed, value)
                step = np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(11),
                                       -jacobian.T @ value)
                candidate = parameters + step
                # A camera slapped onto the wrist must remain close to it.
                candidate[3:6] = np.clip(candidate[3:6], -0.5, 0.5)
                candidate[6:9] = np.clip(candidate[6:9], -10., 10.)
                candidate[9:11] = np.clip(candidate[9:11], math.log(math.tan(math.radians(5))),
                                         math.log(math.tan(math.radians(75))))
                newer = residual(candidate, seed)
                if newer @ newer < value @ value:
                    parameters = candidate
                    damping = max(1e-8, damping / 3.)
                    if np.linalg.norm(step) < 1e-7:
                        break
                else:
                    damping = min(1e6, damping * 10.)
            value = residual(parameters, seed)
            score = float(value @ value)
            if best is None or score < best[0]:
                best = score, parameters, seed
    _, parameters, seed = best
    value = residual(parameters, seed)
    singular = np.linalg.svd(derivative(parameters, seed, value), compute_uv=False)
    if singular[-1] < 1e-4 or singular[0] / singular[-1] > 1e5:
        raise ValueError("Mount is not observable yet; add poses with different wrist AND arm angles")
    _, camera, mount = predict(parameters, seed)
    if np.any(camera[:, 2] < 0.2) or np.any(camera[:, 2] > 10.):
        raise ValueError("Calibration produced invalid target depth; keep one stationary target")
    fitted_scales = np.exp(parameters[9:11])
    errors = camera[:, :2] / camera[:, 2, None] / fitted_scales - pixels
    rms = float(np.sqrt(np.mean(errors[~training] ** 2)))
    if rms > 0.025 or np.max(np.abs(errors)) > 0.075:
        raise ValueError(f"Calibration validation error {rms:.3f}; target may have moved or FOV needs adjusting")
    if np.linalg.norm(parameters[3:6]) > 0.5:
        raise ValueError("Fitted mount is too far from the link; add varied poses and check camera_link")
    return {
        "camera_right": mount[:, 0].tolist(),
        "camera_down": mount[:, 1].tolist(),
        "camera_offset": parameters[3:6].tolist(),
        "estimated_depth_m": float(np.median(camera[:, 2])),
        "horizontal_fov_deg": float(np.degrees(2 * np.arctan(fitted_scales[0]))),
        "vertical_fov_deg": float(np.degrees(2 * np.arctan(fitted_scales[1]))),
    }, rms
