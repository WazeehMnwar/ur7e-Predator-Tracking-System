"""Recognize a deliberate BOSS open-palm command from MediaPipe landmarks."""

from math import acos, degrees, hypot, sqrt


def _distance(a, b):
    return hypot(a.x - b.x, a.y - b.y)


def _angle(a, b, c):
    first = (a.x - b.x, a.y - b.y, a.z - b.z)
    second = (c.x - b.x, c.y - b.y, c.z - b.z)
    length = sqrt(sum(value * value for value in first)) * sqrt(
        sum(value * value for value in second)
    )
    if length == 0:
        return 0.0
    cosine = sum(x * y for x, y in zip(first, second)) / length
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def open_palm_issue(landmarks, world_landmarks, handedness):
    """Return the first reason a hand is not a clear, frontal open palm."""
    if len(landmarks) != 21 or len(world_landmarks) != 21:
        return "HAND LANDMARKS INCOMPLETE"
    if handedness not in ("Left", "Right"):
        return "HAND POSE UNCERTAIN"

    wrist, index_mcp, pinky_mcp = (landmarks[i] for i in (0, 5, 17))
    palm_width = _distance(index_mcp, pinky_mcp)
    if palm_width < 0.035:
        return "MOVE HAND CLOSER TO CAMERA"

    a, b, c = (world_landmarks[i] for i in (0, 5, 17))
    u = (b.x - a.x, b.y - a.y, b.z - a.z)
    v = (c.x - a.x, c.y - a.y, c.z - a.z)
    normal = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    normal_length = sqrt(sum(value * value for value in normal))
    if normal_length == 0 or abs(normal[2]) / normal_length < 0.40:
        return "FACE OPEN PALM TOWARD CAMERA"

    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12),
                               (13, 14, 15, 16), (17, 18, 19, 20)):
        if _angle(world_landmarks[mcp], world_landmarks[pip],
                  world_landmarks[dip]) < 140:
            return "SPREAD ALL FIVE FINGERS"
        if _angle(world_landmarks[pip], world_landmarks[dip],
                  world_landmarks[tip]) < 135:
            return "SPREAD ALL FIVE FINGERS"
        if _distance(landmarks[tip], wrist) < 1.04 * _distance(
            landmarks[pip], wrist
        ):
            return "SPREAD ALL FIVE FINGERS"

    if _angle(world_landmarks[2], world_landmarks[3], world_landmarks[4]) < 130:
        return "SPREAD ALL FIVE FINGERS"
    if _distance(landmarks[4], index_mcp) < 0.55 * palm_width:
        return "SPREAD ALL FIVE FINGERS"
    if _distance(landmarks[4], pinky_mcp) < 0.95 * palm_width:
        return "SPREAD ALL FIVE FINGERS"
    return None


def is_open_palm(landmarks, world_landmarks, handedness):
    return open_palm_issue(landmarks, world_landmarks, handedness) is None


def boss_palm_owner(result, people, verified_boss_track_ids, width, height,
                    boss_track_ids=None):
    """Return (BOSS track, landmarks, status) for a detected hand."""
    if boss_track_ids is None:
        boss_track_ids = verified_boss_track_ids
    issue = "HAND SEEN - OPEN PALM TO CAMERA"
    for index, landmarks in enumerate(result.hand_landmarks):
        if index >= len(result.handedness) or index >= len(result.hand_world_landmarks):
            issue = "HAND SEEN - WAIT FOR HAND LANDMARKS"
            continue
        categories = result.handedness[index]
        if not categories or categories[0].score < 0.5:
            issue = "HAND SEEN - HOLD HAND STEADY"
            continue
        pose_issue = open_palm_issue(
            landmarks, result.hand_world_landmarks[index],
            categories[0].category_name,
        )
        if pose_issue is not None:
            issue = "HAND SEEN - " + pose_issue
            continue

        if not boss_track_ids:
            issue = "HAND SEEN - CLICK YOURSELF, PRESS B"
            continue

        wrist = (landmarks[0].x * width, landmarks[0].y * height)
        palm = (
            sum(landmarks[i].x for i in (0, 5, 9, 13, 17)) * width / 5,
            sum(landmarks[i].y for i in (0, 5, 9, 13, 17)) * height / 5,
        )
        owners = []
        for person in people:
            x1, y1, x2, y2 = person["box"]
            pad_x = max(24, (x2 - x1) * 0.20)
            pad_y = max(24, (y2 - y1) * 0.12)
            if (x1 - pad_x <= wrist[0] <= x2 + pad_x
                    and y1 - pad_y <= wrist[1] <= y2 + pad_y
                    and x1 - pad_x <= palm[0] <= x2 + pad_x
                    and y1 - pad_y <= palm[1] <= y2 + pad_y):
                owners.append(person["track_id"])
        if len(owners) == 1:
            owner_id = owners[0]
            if owner_id in verified_boss_track_ids:
                return owner_id, landmarks, None
            if owner_id in boss_track_ids:
                issue = "BOSS SEEN - SHOW FACE TO CAMERA"
            else:
                issue = "HAND SEEN - HAND BELONGS TO ANOTHER PERSON"
        elif len(owners) > 1:
            issue = "HAND SEEN - PEOPLE TOO CLOSE TOGETHER"
        else:
            issue = "HAND SEEN - MOVE HAND NEAR BOSS"
    return None, None, issue


def hand_boxes(result, width, height, padding=8):
    """Pixel boxes for every detected hand, regardless of gesture or owner."""
    boxes = []
    for landmarks in result.hand_landmarks:
        xs = [int(point.x * width) for point in landmarks]
        ys = [int(point.y * height) for point in landmarks]
        boxes.append((
            max(0, min(xs) - padding),
            max(0, min(ys) - padding),
            min(width - 1, max(xs) + padding),
            min(height - 1, max(ys) + padding),
        ))
    return boxes


class PalmHoldDetector:
    def __init__(self, hold_seconds=0.5, min_frames=4, cooldown_seconds=5.0):
        self.hold_seconds = hold_seconds
        self.min_frames = min_frames
        self.cooldown_seconds = cooldown_seconds
        self.owner_id = None
        self.first_seen = None
        self.frames = 0
        self.latched = False
        self.cooldown_until = 0.0
        self.release_started = None
        self.release_frames = 0

    def update(self, now, owner_id):
        if owner_id is None:
            self.owner_id = None
            self.first_seen = None
            self.frames = 0
            self.release_started = now if self.release_started is None else self.release_started
            self.release_frames += 1
            if self.release_frames >= 3 and now - self.release_started >= 0.25:
                self.latched = False
            return False

        self.release_started = None
        self.release_frames = 0
        if owner_id != self.owner_id:
            self.owner_id = owner_id
            self.first_seen = now
            self.frames = 1
        else:
            self.frames += 1

        if (not self.latched and now >= self.cooldown_until
                and self.frames >= self.min_frames
                and now - self.first_seen >= self.hold_seconds):
            self.latched = True
            self.cooldown_until = now + self.cooldown_seconds
            return True
        return False
