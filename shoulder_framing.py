"""Bounded background pose inference for the selected person's shoulders."""
import math
import threading
import time


def shoulder_width(landmarks, world, crop, image_width):
    """Return full-image width fraction; reject occlusion and sideways poses."""
    if len(landmarks) < 13 or len(world) < 13 or image_width <= 0:
        return 0., False
    a, b = landmarks[11], landmarks[12]
    for p in (a, b):
        if not all(math.isfinite(v) for v in (p.x, p.y, p.visibility, p.presence)):
            return 0., False
        if min(p.visibility, p.presence) < .7 or not (.03 < p.x < .97 and .03 < p.y < .97):
            return 0., False
    wa, wb = world[11], world[12]
    delta = [getattr(wa, k)-getattr(wb, k) for k in ('x', 'y', 'z')]
    length = math.sqrt(sum(v*v for v in delta))
    if not math.isfinite(length) or length < .05 or abs(delta[2])/length > .5:
        return 0., False
    x1, y1, x2, y2 = crop
    dx, dy = abs(a.x-b.x)*(x2-x1), abs(a.y-b.y)*(y2-y1)
    if dx < 12 or dy > dx*.5:
        return 0., False
    width = math.hypot(dx, dy)/image_width
    return (width, True) if 0 < width < 1 else (0., False)


class ShoulderFraming:
    """One pending crop at most; slow inference never queues old camera frames."""
    def __init__(self, model_path):
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.pending = None
        self.result = None
        self.identity = None
        self.generation = 0
        self.last_submit = -math.inf
        self.closed = False
        self.worker = None
        try:
            import mediapipe as mp
            self.mp = mp
            self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(
                mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_poses=1, min_pose_detection_confidence=.6,
                    min_pose_presence_confidence=.7))
        except Exception as exc:
            print(f'Shoulder reach unavailable: {exc}. Pointing remains available.')
            return
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def observe(self, frame, person, captured_at):
        import cv2
        identity = None if person is None else int(person['track_id'])
        with self.lock:
            if identity != self.identity:
                self.identity = identity
                self.generation += 1
                self.result = self.pending = None
            generation = self.generation
        if identity is None or self.worker is None:
            return None
        now = time.monotonic()
        if now-self.last_submit >= .15:
            self.last_submit = now
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = map(int, person['box'])
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(width,x2), min(height,y2)
            if x2 > x1 and y2 > y1 and person.get('confidence', 0) >= .6:
                crop = frame[y1:y2, x1:x2]
                scale = min(1., 384/max(crop.shape[:2]))
                crop = cv2.resize(crop, (max(1,round(crop.shape[1]*scale)), max(1,round(crop.shape[0]*scale))))
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.pending = (rgb, (x1,y1,x2,y2), width, identity, generation, captured_at)
                self.event.set()
            else:
                with self.lock:
                    self.result = (0., identity, False, captured_at)
        with self.lock:
            return self.result

    def _run(self):
        while True:
            self.event.wait()
            with self.lock:
                self.event.clear()
                if self.closed:
                    break
                item, self.pending = self.pending, None
            if item is None:
                continue
            rgb, crop, width, identity, generation, captured_at = item
            if time.monotonic()-captured_at > .35:
                continue
            try:
                result = self.pose.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
                value, valid = shoulder_width(result.pose_landmarks[0], result.pose_world_landmarks[0], crop, width) if result.pose_landmarks else (0., False)
            except Exception as exc:
                print(f'Shoulder inference failed: {exc}')
                value, valid = 0., False
            with self.lock:
                if generation == self.generation:
                    self.result = (value, identity, valid, captured_at)
        self.pose.close()

    def close(self):
        with self.lock:
            self.closed = True
        self.event.set()
        if self.worker is not None:
            self.worker.join(timeout=2.)
