"""Small runtime helpers for a live camera pipeline."""

import threading
import time


class LatestFrameCapture:
    """Drain a camera in one thread and expose only its newest frame."""

    def __init__(self, capture, on_frame=None):
        self.capture = capture
        self.on_frame = on_frame
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.captured_at = None
        self.stopping = False
        self.failed = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _read_loop(self):
        while True:
            with self.condition:
                if self.stopping:
                    return
            ok, frame = self.capture.read()
            captured_at = time.monotonic()
            if ok and self.on_frame is not None:
                try:
                    self.on_frame(frame, captured_at)
                except Exception:
                    with self.condition:
                        self.failed = True
                        self.condition.notify_all()
                    raise
            with self.condition:
                if self.stopping:
                    return
                if not ok:
                    self.failed = True
                    self.condition.notify_all()
                    return
                self.frame = frame
                self.sequence += 1
                self.captured_at = captured_at
                self.condition.notify_all()

    def read_latest(self, after_sequence=0, timeout=2.0):
        """Wait for a new frame; intermediate camera frames are discarded."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > after_sequence or self.failed or self.stopping,
                timeout=timeout,
            )
            if self.sequence <= after_sequence:
                return False, None, self.sequence, self.captured_at
            return True, self.frame, self.sequence, self.captured_at

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.thread.join(timeout=1.0)
        self.capture.release()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class PeriodicTask:
    """Allow a task at most once per elapsed interval."""

    def __init__(self, interval_seconds):
        self.interval_seconds = interval_seconds
        self.last_run = None

    def due(self, now):
        if (self.last_run is None
                or now - self.last_run >= self.interval_seconds - 1e-9):
            self.last_run = now
            return True
        return False


def shifted_recent_face(cached, body_box, now, max_age_seconds):
    """Keep a recent face point attached to its moving person box."""
    if cached is None:
        return None
    face, previous_box, seen_at = cached
    if now - seen_at > max_age_seconds:
        return None
    shifted = face.copy()
    shifted[0] += (body_box[0] + body_box[2]
                   - previous_box[0] - previous_box[2]) / 2
    shifted[1] += (body_box[1] + body_box[3]
                   - previous_box[1] - previous_box[3]) / 2
    return shifted


def needs_face_feature(identity, role, last_verified_at, now, verify_interval):
    """Embed unknown faces and periodically recheck the BOSS face."""
    return identity is None or (
        role == "boss" and (
            last_verified_at is None or now - last_verified_at >= verify_interval
        )
    )


def matching_hand_result(outputs, frame_timestamp_ms, max_age_ms):
    """Use hand landmarks from this frame or a nearby earlier frame."""
    for result, timestamp_ms in reversed(outputs):
        if timestamp_ms <= frame_timestamp_ms:
            if frame_timestamp_ms - timestamp_ms <= max_age_ms:
                return result, timestamp_ms
            break
    return None, -1
