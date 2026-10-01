"""Body-box framing feedback. Pure proportional reach rate, no integral."""
from dataclasses import dataclass
import math


@dataclass
class ReachConfig:
    metric: str = 'body_height'
    target_width: float = .30
    target_height: float = .45
    deadband: float = .05
    radius_min: float = .261095045791
    radius_max: float = .628725446010
    height_min: float = .74
    height_max: float = .78
    max_speed: float = .005
    gain: float = .04
    smoothing_time: float = .5
    settle_time: float = .7
    max_box_rate: float = .5
    box_jump_floor: float = .03
    center_limit: float = .15
    center_resume: float = .10
    alignment_time: float = 1.0
    resume_time: float = .4
    stale_time: float = .35
    direction: int = 1

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name == 'metric':
                continue
            if not math.isfinite(float(getattr(self,name))):
                raise ValueError(f'Invalid reach.{name}')
        if not (self.metric in ('body_height', 'shoulder_width')
                and 0 < self.deadband < self.target_width < 1
                and 0 < self.target_height < 1 and 0 < self.deadband < self.target_height
                and 0 < self.radius_min < self.radius_max < 1.5
                and 0 < self.height_min < self.height_max < 2
                and 0 < self.max_speed <= .02 and 0 < self.gain <= .2
                and .1 <= self.smoothing_time <= 3 and .2 <= self.settle_time <= 5
                and 0 < self.max_box_rate <= 2 and 0 < self.center_limit <= .3
                and 0 <= self.box_jump_floor <= .1
                and 0 < self.center_resume < self.center_limit
                and .2 <= self.alignment_time <= 5 and .1 <= self.resume_time <= 5
                and 0 < self.stale_time <= .35 and self.direction in (-1,1)):
            raise ValueError('Invalid reach settings')


class ReachControl:
    def __init__(self, config):
        self.config=config
        self.last_received=None
        self.last_height=None
        self.last_valid=False
        self.received_count=0
        self.reset()

    def reset(self):
        self.identity=None
        self.at=None
        self.raw=None
        self.filtered=None
        self.stable_since=None
        self.reset_alignment()

    def reset_alignment(self):
        self.aligned_since=None
        self.last_alignment_at=None
        self.acquired=False
        self.allowed=False
        self.reason='waiting for stable detection'

    def pause(self, reason, reacquire=False):
        if reacquire:
            self.reset_alignment()
        self.allowed=False
        self.aligned_since=None
        self.reason=reason
        return 0.

    def observe(self, height, identity, valid, observed_at):
        c=self.config
        self.last_received=observed_at
        self.last_height=height
        self.last_valid=valid
        self.received_count+=1
        if (valid not in (1,2) or not math.isfinite(height) or not 0 < height <= 1
                or not math.isfinite(observed_at)):
            self.reset(); return
        if self.at is not None and observed_at <= self.at:
            return
        if self.at is None or identity != self.identity or observed_at-self.at > c.stale_time:
            self.reset_alignment()
            self.identity=identity; self.at=observed_at
            self.raw=self.filtered=height; self.stable_since=observed_at
            return
        dt=observed_at-self.at
        # Rate alone mistakes ordinary detector jitter for occlusion at high
        # FPS. Require a meaningful size jump as well as a high rate.
        if abs(height-self.raw) > max(c.box_jump_floor, c.max_box_rate*dt):
            self.reset_alignment()
            self.filtered=height
            self.stable_since=observed_at
        else:
            self.filtered += (1-math.exp(-dt/c.smoothing_time))*(height-self.filtered)
        self.at=observed_at; self.raw=height

    def rate(self, now, error, observed_at=None):
        c=self.config
        observed_at = now if observed_at is None else observed_at
        if (not math.isfinite(now) or not math.isfinite(observed_at)
                or len(error) != 2 or not all(math.isfinite(float(v)) for v in error)):
            return self.pause('invalid alignment', reacquire=True)
        if (self.at is None or now-self.at > c.stale_time or now < self.at
                or now-observed_at > c.stale_time or observed_at > now):
            return self.pause('stale detection', reacquire=True)
        if now-self.stable_since < c.settle_time:
            return self.pause('waiting for stable box', reacquire=True)
        if (self.last_alignment_at is not None
                and (observed_at < self.last_alignment_at
                     or observed_at-self.last_alignment_at > c.stale_time)):
            self.reset_alignment()
        self.last_alignment_at=observed_at
        center_error=max(abs(float(v)) for v in error)
        if center_error > c.center_limit:
            return self.pause('pointing has priority')
        if not self.allowed:
            if center_error > c.center_resume:
                return self.pause('waiting for tighter alignment')
            if self.aligned_since is None:
                self.aligned_since=observed_at
            wait=c.resume_time if self.acquired else c.alignment_time
            if observed_at-self.aligned_since < wait:
                self.reason='settling alignment'
                return 0.
            self.allowed=True
            self.acquired=True
        target = c.target_width if c.metric == 'shoulder_width' else c.target_height
        error=target-self.filtered
        correction=math.copysign(max(0.,abs(error)-c.deadband),error)
        if self.last_valid == 2 and correction >= 0:
            self.reason='vertically clipped body: extension blocked'
            return 0.
        self.reason='framing satisfied' if correction == 0 else 'active'
        return c.direction*max(-c.max_speed,min(c.max_speed,c.gain*correction))

    def diagnostic(self, now):
        if self.config.metric == 'shoulder_width':
            if self.last_received is None:
                return 'no SHOULDERS packets: update target_tracker.py and shoulder_framing.py'
            if now-self.last_received > self.config.stale_time:
                return 'shoulder data stale'
            if not self.last_valid:
                return 'shoulders unavailable, occluded, or sideways: reach paused'
            return self.reason
        if self.last_received is None:
            return 'no BOX packets: update the running target_tracker.py'
        if now-self.last_received > self.config.stale_time:
            return f'BOX data stale ({now-self.last_received:.2f}s)'
        if not self.last_valid:
            return 'body box rejected: confidence below 0.6 or invalid geometry'
        return self.reason + ('; vertically clipped, retract only' if self.last_valid == 2 else '')
