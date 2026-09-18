"""Bounded target-loss recovery for the FB bridge; no independent publisher."""
from collections import deque
from dataclasses import dataclass
import math


SEARCH_STATES = {'LOST_COAST', 'SEARCH_HOME', 'SEARCH_SETTLE', 'SEARCH_PAN', 'SEARCH_PAUSE'}


@dataclass
class SearchConfig:
    enabled: bool = True
    edge_threshold: float = .65
    min_image_speed: float = .08
    coast_time: float = .5
    coast_speed: float = .03
    half_span: float = math.pi/4
    settle_time: float = .5
    pause_total: float = 5.
    cycles: int = 2
    timeout: float = 180.
    heartbeat_timeout: float = 1.5

    def __post_init__(self):
        if not isinstance(self.enabled,bool): raise ValueError('search.enabled must be boolean')
        bounds={'edge_threshold':(.3,.95),'min_image_speed':(.01,2.),
                'coast_time':(0.,1.),'coast_speed':(.005,.05),'half_span':(.01,math.pi/4),
                'settle_time':(.2,3.),'pause_total':(1.,15.),'timeout':(10.,300.),
                'heartbeat_timeout':(.35,3.)}
        for k,(lo,hi) in bounds.items():
            v=float(getattr(self,k))
            if not math.isfinite(v) or not lo <= v <= hi: raise ValueError(f'invalid search.{k}')
            setattr(self,k,v)
        if isinstance(self.cycles,bool) or not isinstance(self.cycles,int) or not 1 <= self.cycles <= 3:
            raise ValueError('search.cycles must be 1..3')


class TargetSearch:
    def search_init(self, config):
        self.search_config=config
        self.target_history=deque(maxlen=30)
        self.missing_at=None
        self.search_started=None
        self.search_index=0
        self.search_goal=None
        self.manual_stop=False
        self.servo_halted=False
        self.search_armed=False

    def record_track(self, error, at):
        if self.target_history and at <= self.target_history[-1][0]: return
        if self.target_history and at-self.target_history[-1][0] > .35:
            self.target_history.clear()
        self.target_history.append((at,float(error[0])))
        if not self.manual_stop and not self.servo_halted and not self.output_fault:
            self.search_armed=True

    def cancel_search(self):
        self.search_armed=False
        self.target_history.clear()
        self.search_started=None
        self.missing_at=None

    def missing_target(self, now, observed_at):
        if not 0 <= now-observed_at <= .35:
            return
        self.missing_at=observed_at
        if (self.state not in ('TRACK', 'STOP') or not self.search_armed
                or not self.search_config.enabled or self.manual_stop
                or self.servo_halted or self.output_fault or not self.start_complete
                or self.model is None or self.positions is None or now-self.joints_at > .3
                or not self.target_history):
            if self.state == 'TRACK': self.stop(); self.halt()
            return
        c=self.search_config
        self.search_armed=False
        margin=self.tracking_config.joint_margin
        if any(not self.model.lower[i]+margin <= q <= self.model.upper[i]-margin
               for i,q in enumerate(self.start_pose)):
            self.stop(); self.halt()
            self.get_logger().error('Search START exceeds robot joint limits')
            return
        self.search_started=now
        self.search_index=0
        self.search_goal=None
        self.track=None
        self.halt()
        latest=self.target_history[-1]
        history=[p for p in self.target_history if .1 <= latest[0]-p[0] <= .4]
        direction=0.
        if (now-latest[0] <= .35 and history and abs(latest[1]) >= c.edge_threshold
                and self.ik_config.pan_sign is not None):
            old=history[0]; trend=(latest[1]-old[1])/(latest[0]-old[0])
            if latest[1]*trend > 0 and abs(trend) >= c.min_image_speed:
                direction=math.copysign(1.,latest[1])*self.ik_config.pan_sign
        if direction and c.coast_time > 0:
            self.coast_direction=direction
            self.set_state('LOST_COAST'); self.phase_at=now
            self.get_logger().info('Target left image edge: brief bounded pan follow')
        else:
            self.set_state('SEARCH_HOME')
        low,high=self.search_pan_bounds()
        self.get_logger().info(
            f'Search scheduled: return to START, then base {math.degrees(low):.1f} to '
            f'{math.degrees(high):.1f} deg, {c.cycles} cycles')

    def search_pan_bounds(self):
        margin=self.tracking_config.joint_margin
        span=min(self.search_config.half_span,self.ik_config.pan_span)
        return (max(self.start_pose[0]-span,self.model.lower[0]+margin),
                min(self.start_pose[0]+span,self.model.upper[0]-margin))

    def search_pose_arrived(self, now):
        self.halt()
        if self.state == 'SEARCH_HOME':
            if self.track is not None and now-self.track_at <= .35:
                self.set_state('TRACK')
            else:
                self.set_state('SEARCH_SETTLE'); self.phase_at=now
        elif self.state == 'SEARCH_PAN':
            self.set_state('SEARCH_PAUSE'); self.phase_at=now

    def run_search(self, now, dt):
        c=self.search_config
        # Missing-target packets are a heartbeat. Network/camera failure is STOP.
        heartbeat=max(self.missing_at if self.missing_at is not None else -math.inf,
                      self.track_at if self.track is not None else -math.inf)
        reason = None
        if now-heartbeat > c.heartbeat_timeout:
            reason = f'vision heartbeat absent for {now-heartbeat:.2f}s'
        elif self.model is None or self.servo_halted:
            reason = 'robot model unavailable or Servo halted'
        elif self.search_started is None or now-self.search_started > c.timeout:
            reason = 'search duration exhausted'
        if reason:
            self.stop(); self.halt()
            self.get_logger().warn(f'Search stopped: {reason}')
            return
        if now-heartbeat > .35:
            # Stop outputs immediately, but preserve the search phase through a
            # brief inference/network stall. No blind motion during the gap.
            self.halt()
            return
        if self.state == 'SEARCH_HOME':
            self.run_pose(now,self.start_pose,range(6))
        elif self.state == 'LOST_COAST':
            if now-self.phase_at >= c.coast_time:
                self.halt(); self.set_state('SEARCH_HOME'); return
            # Only base pan. Retain normal speed, acceleration and position limits.
            low=max(self.start_pose[0]-self.ik_config.pan_span,
                    self.model.lower[0]+self.tracking_config.joint_margin)
            high=min(self.start_pose[0]+self.ik_config.pan_span,
                     self.model.upper[0]-self.tracking_config.joint_margin)
            speed=min(c.coast_speed,self.tracking_config.max_joint_speed,self.model.speed[0])
            desired=self.coast_direction*speed
            lower=max(-speed,min(0.,(low-self.positions[0])/.5))
            upper=min(speed,max(0.,(high-self.positions[0])/.5))
            previous=max(lower,min(upper,self.last_jog[0]))
            step=self.tracking_config.max_joint_acceleration*dt
            velocity=max(max(lower,previous-step),min(min(upper,previous+step),desired))
            self.last_jog=(velocity,0.,0.,0.,0.,0.)
            self.publish_twist(); self.publish_jog(self.last_jog,(0,))
        elif self.state in ('SEARCH_SETTLE','SEARCH_PAUSE'):
            self.halt()
            delay=c.settle_time if self.state == 'SEARCH_SETTLE' else c.pause_total/3
            if now-self.phase_at < delay: return
            if self.state == 'SEARCH_PAUSE': self.search_index+=1
            if self.search_index >= c.cycles*3:
                self.stop(); return
            low,high=self.search_pan_bounds()
            self.search_goal=list(self.start_pose)
            self.search_goal[0]=(low,high,self.start_pose[0])[self.search_index%3]
            self.set_state('SEARCH_PAN')
        elif self.state == 'SEARCH_PAN':
            self.run_pose(now,self.search_goal,(0,))
