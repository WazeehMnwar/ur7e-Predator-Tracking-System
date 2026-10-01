# FB framing test

The original arm bridge and original pose YAML are unchanged. Run only one
arm bridge at a time; both receive vision on UDP port 5005.

## Files to transfer

- vision_moveit_servoFB.py
- planar_trackingFB.py
- reach_control.py
- target_search.py
- predator_servo_posesFB.yaml
- whole_arm_tracking.py (existing shared URDF helper)
- predator_servo.launch.py (existing launch)
- target_tracker.py (updated to additionally send BODY-box height/quality)

Face detection, selection, memory and gestures were not changed. The original
arm bridge ignores the new BOX packets. An older vision script supplies no BOX
packets: pointing still works, but reach stays paused.

## Run

Start the existing UR driver and External Control as usual. In separate ROS
terminals, from the directory containing these files:

```bash
ros2 launch ./predator_servo.launch.py poses_file:=./predator_servo_posesFB.yaml
```

```bash
/usr/bin/python3 vision_moveit_servoFB.py --enable-motion --enable-reach
```

Start `target_tracker.py` in the usual vision environment and select a target.
The bridge runs START before tracking. Omitting --enable-reach disables box-size
movement. Do not run vision_moveit_servo.py alongside the FB bridge.

To disable further reach adjustments without stopping pointing, run on the
arm computer:

```bash
python3 -c 'import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"REACH_OFF", ("127.0.0.1", 5005))'
```

This freezes the radius target; bounded joint tracking may finish settling to
that target. Restart with --enable-reach to enable it again. STOP/target loss
halts all tracking commands. Bow preserves and restores the live pose.

## Defaults and geometry

- Target body-box height: 45% of image height.
- Hold band: 40–50%.
- Maximum requested radius-target rate: 0.00625 m/s (6.25 mm/s).
- FB speeds/gains increased 25%: joint speed cap 0.125 rad/s, pose speed cap
  0.100 rad/s, pose gain 0.75, pointing gain 0.15 and pose-following gain 1.25.
  Acceleration remains 0.20 rad/s². Actual motion still depends on image error,
  IK and Servo limits, so this is not a guarantee of exactly 25% faster travel.
  Restart Servo with the FB YAML so its output cap matches the bridge.
- No accumulated box-size integral. Radius-target integration is bounded, and
  pauses while the actual arm lags behind its joint target.
- Body-height smoothing: 0.5 s. After target acquisition/size jumps, require
  0.7 s of stable observations, followed by 1.0 s of continuously good alignment.
- Reach starts below 0.10 normalized centering error on BOTH image axes;
  pauses above 0.15 on EITHER axis. After an ordinary centering pause, resume
  only after 0.4 s continuously below 0.10. Errors between 0.10 and 0.15 keep
  an already-active reach loop running but cannot start a paused one.
- Alignment timing uses fresh vision observation timestamps, not repeated
  control ticks on the same image. Target changes, invalid/clipped boxes,
  stale input and abrupt size jumps require full reacquisition.
- Pointing stays active during these reach pauses. Terminal reach status
  distinguishes waiting for stable detection, settling alignment, pointing
  priority, framing satisfied, and active corrections.
- Freeze for centering error over 0.15 normalized units, stale data over 0.35 s,
  confidence below 0.6, boxes touching the outer 2% of frame, and sudden box
  height rates above 0.5 image-heights/s. Gradual crouching can still fool this
  cue; bounding-box framing is not a distance sensor.

Nominal UR7e tool0 FK, assuming wrist 2/3 = 90/-90 degrees in the cropped images:

| Pose | Joint angles (degrees, UR order) | Radius (m) | Height (m) |
|---|---|---|---|
| Rear screenshot | -70,-40,-49,-96,90,-90 | 0.261095 | 0.918465 |
| Forward screenshot | -70,-134,9,-52,90,-90 | 0.628725 | 0.894267 |
| START | -90,-50,-105,-21,90,-90 | 0.219725 | 0.760225 |

The receiver also prints screenshot FK using its running robot model. YAML
contains the nominal-derived limits; inspect the printout for model differences.
These are tool0 coordinates, not an unmodeled lens.

The screenshots select opposite elbow branches. Only their radii are used:
the controller keeps START's negative-elbow branch and fixed START height.
The 0.74–0.78 m height guard allows START height and rejects configurations
outside that interval; no autonomous height changes are requested.

START is below the rear radius bound. It will not jump to the bound. Reach
can extend slowly toward the interval, but cannot retract farther while below
it. Once inside, reach stays within 0.261095–0.628725 m. Full screenshot radius
range passed nominal IK sampling at START height/pitch on the retained branch.
Other pitches can reduce the feasible range. Infeasible steps are reduced or
held; pointing is retried without reach if the combined step fails.

The direction setting assumes increasing radius approaches the person and
grows their box. That depends on the camera mount and working direction; if
box size changes oppositely during extension, disable reach and correct the
mapping. The robot model cannot establish that optical relationship by itself.

Joint speed/acceleration limits and Servo collision checks remain active.
Radius-target speed is not a certified bound on actual Cartesian speed through
Servo filtering or pose transients. Endpoint FK/IK tests do not prove workspace
clearance or physical camera convergence. Physical movement has not been tested.

Offline checks (no motion):

```bash
python3 -m unittest test_reach_control test_planar_tracking test_whole_arm_tracking -q
```

## Lost target and idle search

Also transfer `target_search.py`, the updated FB bridge/YAML, and updated
`target_tracker.py`. Detection and face memory algorithms are unchanged.

- Vision sends TARGET_MISSING only while a target is still selected but absent.
  TARGET_LOST means deliberate clearing or gesture suppression and never starts
  a search. Older vision versions sending TARGET_LOST do not enable searching.
- A recent outward horizontal trend near an image edge permits at most 0.5 s
  of slow base-only following (0.0375 rad/s, acceleration/position limited).
  Center occlusion skips this step. No reach motion is requested during search.
- Return to START using the existing pose controller; settle 0.5 s; then move
  only shoulder pan: left, pause, right, pause, center, pause. Two cycles then
  hold at START until the selected person is visible again.
- Requested span is +/-45 degrees but clipped to the existing pan limits.
  Current ik.pan_span=0.7 permits +/-40.1 degrees, about 80.2 degrees total.
  Five seconds is total pause time PER CYCLE, in addition to travel time.
  Travel uses the 0.10 rad/s pose speed cap; a full cycle takes tens of seconds.
- Reacquisition during a sweep/coast resumes pointing. During the return to
  START it finishes that return first, then tracks if the observation is fresh.
  Reach must pass its stable-detection/alignment gates again.
- Vision heartbeat gaps over 0.35 s immediately hold outputs at zero; search
  keeps its state and resumes on fresh vision. Gaps exceeding 1.5 s cancel the
  search. Tracking uses the same hold/recovery distinction, preventing brief
  processing stalls from repeatedly switching TRACK to STOP and back.
  Stale joint data, timeouts, explicit STOP, faults and Servo halts stop searching.
  Explicit STOP and Servo halts latch a stop;
  after Servo clears, send START or restart the bridge to permit motion again.
  Search never starts merely because nobody was selected on launch.
- BOW can interrupt search; it restores its saved pose and does not restart
  an old search automatically. START/BOW paths remain velocity-controlled,
  not preplanned collision-free trajectories. Physical clearance is unverified.

Disable autonomous search using `search.enabled: false` in the FB YAML.
Search expires after 180 s total; each reused pose move retains its 60 s timeout.
A timeout holds the current position rather than forcing a return.

Offline search tests: `python3 -m unittest test_target_search -q`.

## Diagnosing absent FB motion

The provided Jetson log launched with `--enable-motion` only: `reach=False`
means reach was disabled. Add `--enable-reach`; it remains opt-in.
`body_height=none` alone cannot distinguish missing packets from rejected body
boxes. Updated status reports `no BOX packets`, `BOX data stale`, or
`body box rejected` (clipping/low confidence), even when reach is disabled.
Run the current target_tracker.py and keep the whole person inside the image
to supply a usable body height. The target is BODY height, not face height.

The logs also showed SEARCH_HOME stopping before its scan with a combined
heartbeat/model/timeout message. It was too early for the total search timeout;
brief vision gaps are the likely cause. Updated search logs name the specific
stop cause and hold still through short gaps instead of abandoning the search.
Neither search recovery nor speed tuning bypasses travel/joint limits.
