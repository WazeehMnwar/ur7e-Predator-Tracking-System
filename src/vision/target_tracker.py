#!/usr/bin/env python3

import json
import os
import socket
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from boss_gesture import PalmHoldDetector, boss_palm_owner, hand_boxes
from body_framing import body_height_signal
from shoulder_framing import ShoulderFraming
from vision_runtime import (
    LatestFrameCapture, PeriodicTask, matching_hand_result,
    needs_face_feature, shifted_recent_face,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]  # repo root (two levels up from src/vision/)
MODELS_DIR = ROOT_DIR / "models"

shoulder_framing = ShoulderFraming(MODELS_DIR / 'hand/pose_landmarker_lite.task')
YOLO_MODEL = os.environ.get("PREDATOR_YOLO_MODEL", str(MODELS_DIR / "yolo/yolo26n.pt"))
YOLO_INPUT_SIZE = int(os.environ.get("PREDATOR_YOLO_SIZE", "512"))

YUNET_MODEL = str(MODELS_DIR / "face/face_detection_yunet_2023mar.onnx")
SFACE_MODEL = str(MODELS_DIR / "face/face_recognition_sface_2021dec.onnx")
# Official MediaPipe model bundle: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
HAND_MODEL = str(MODELS_DIR / "hand/hand_landmarker.task")

CAMERA_ID = int(os.environ.get("PREDATOR_CAMERA_ID", "0"))
WINDOW_NAME = "Predator Vision"
ARM_UDP_HOST = os.environ.get("PREDATOR_ARM_HOST", "127.0.0.1")
ARM_UDP_PORT = int(os.environ.get("PREDATOR_ARM_PORT", "5005"))
arm_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Official OpenCV SFace cosine threshold.
FACE_MATCH_THRESHOLD = 0.363

# New identities are not created from one bad frame.
# We average a few face embeddings first, then check memory again.
AUTO_ENROLL_FRAMES = 5

# Persistent identity database (lives at repo root, never committed).
MEMORY_FILE = ROOT_DIR / "predator_face_memory.json"
LEGACY_MEMORY_FILE = ROOT_DIR / "face_memory.npz"

# When the selected target's face is temporarily not visible but
# BoT-SORT still tracks the body, use an approximate upper-body/head
# location so the target still has a tracking point.
BODY_FALLBACK = True

PALM_HOLD_SECONDS = 0.5
PALM_MIN_FRAMES = 4
PALM_COOLDOWN_SECONDS = 5.0
BOSS_FACE_MAX_AGE_SECONDS = 3.0
COMMAND_DISPLAY_SECONDS = 2.0
HAND_INPUT_MAX_WIDTH = 640
HAND_SUBMIT_INTERVAL_SECONDS = 0.10
HAND_RESULT_MAX_AGE_MS = 450
FACE_DETECT_INTERVAL_SECONDS = 0.20
FACE_CACHE_MAX_AGE_SECONDS = 0.60
BOSS_FACE_VERIFY_INTERVAL_SECONDS = 0.75

hand_output_lock = threading.Lock()
hand_outputs = deque(maxlen=64)


def on_hand_result(result, output_image, timestamp_ms):
    with hand_output_lock:
        hand_outputs.append((result, timestamp_ms))


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO26n...")
yolo = YOLO(YOLO_MODEL)

print("Loading YuNet...")
face_detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

print("Loading SFace...")
face_recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)

print("Loading MediaPipe hands...")
hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=on_hand_result,
    )
)


# ============================================================
# FACE MEMORY
# ============================================================
#
# known_people[label] = {
#     "role": "known", "target", or "boss",
#     "embedding": np.ndarray shape (1, N),
#     "extra_embeddings": list of additional face embeddings
# }
#
# Every person is automatically remembered once.
# There is at most one TARGET and one BOSS.
# ============================================================

known_people = {}


def normalize_feature(feature):
    feature = np.asarray(
        feature,
        dtype=np.float32
    ).reshape(1, -1)

    norm = np.linalg.norm(feature)

    if norm > 0:
        feature = feature / norm

    return feature


def load_memory():
    global known_people

    if not MEMORY_FILE.exists():
        return

    try:
        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            raw = json.load(file)

        for label, record in raw.items():
            known_people[label] = {
                "role": record.get(
                    "role",
                    "known"
                ),
                "embedding": normalize_feature(
                    record["embedding"]
                ),
                "extra_embeddings": [
                    normalize_feature(feature)
                    for feature in record.get("extra_embeddings", [])
                ],
            }

    except Exception as exc:
        print(
            f"WARNING: Could not load memory: {exc}"
        )


def save_memory():
    serializable = {}

    for label, record in known_people.items():
        serializable[label] = {
            "role": record["role"],
            "embedding": (
                record["embedding"]
                .reshape(-1)
                .astype(float)
                .tolist()
            ),
            "extra_embeddings": [
                feature.reshape(-1).astype(float).tolist()
                for feature in record.get("extra_embeddings", [])
            ],
        }

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=MEMORY_FILE.parent,
            prefix=f"{MEMORY_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(serializable, file, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, MEMORY_FILE)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def next_person_label():
    largest = 0

    for label in known_people:
        if not label.startswith("person_"):
            continue

        try:
            number = int(
                label.split("_", 1)[1]
            )

            largest = max(
                largest,
                number
            )

        except ValueError:
            pass

    return f"person_{largest + 1:03d}"


def compare_faces(feature1, feature2):
    return float(
        face_recognizer.match(
            normalize_feature(feature1),
            normalize_feature(feature2),
            cv2.FaceRecognizerSF_FR_COSINE
        )
    )


def recognize_from_memory(feature):
    if feature is None or not known_people:
        return None, -1.0

    best_label = None
    best_score = -1.0

    for label, record in known_people.items():
        score = max(
            compare_faces(saved_feature, feature)
            for saved_feature in [
                record["embedding"],
                *record.get("extra_embeddings", []),
            ]
        )

        if score > best_score:
            best_score = score
            best_label = label

    if (
        best_label is not None
        and best_score >= FACE_MATCH_THRESHOLD
    ):
        return best_label, best_score

    return None, best_score


def enroll_new_identity(feature):
    """
    Create exactly one persistent identity for a genuinely new face.
    """
    # Final duplicate check before creating anything.
    existing_label, score = recognize_from_memory(
        feature
    )

    if existing_label is not None:
        return existing_label, False

    label = next_person_label()

    known_people[label] = {
        "role": "known",
        "embedding": normalize_feature(
            feature
        ),
        "extra_embeddings": [],
    }

    save_memory()

    return label, True


def make_boss(label):
    """
    Exactly one identity can be BOSS.
    BOSS remains a remembered identity but is never a TARGET.
    """

    if label not in known_people:
        return False

    for other_label, record in known_people.items():
        if other_label != label and record["role"] == "boss":
            record["role"] = "known"
    known_people[label]["role"] = "boss"

    save_memory()

    print(
        f"{label} is now BOSS."
    )

    return True


def get_role(label):
    if label is None:
        return None

    if label not in known_people:
        return None

    return known_people[label]["role"]


def make_target(label):
    if label not in known_people:
        return False

    for other_label, record in known_people.items():
        if other_label != label and record["role"] == "target":
            record["role"] = "known"
    known_people[label]["role"] = "target"
    save_memory()
    return True


def import_legacy_memory():
    """Bring older NPZ face samples into the persistent JSON database."""
    if not LEGACY_MEMORY_FILE.exists():
        return

    had_current_memory = bool(known_people)
    changed = False
    try:
        with np.load(LEGACY_MEMORY_FILE, allow_pickle=False) as archive:
            for legacy_label in archive.files:
                feature = normalize_feature(archive[legacy_label])
                label, _ = recognize_from_memory(feature)

                if label is None:
                    label = next_person_label()
                    known_people[label] = {
                        "role": "known",
                        "embedding": feature,
                        "extra_embeddings": [],
                    }
                    changed = True
                else:
                    record = known_people[label]
                    samples = [
                        record["embedding"],
                        *record.get("extra_embeddings", []),
                    ]
                    if not any(compare_faces(saved, feature) >= 0.995
                               for saved in samples):
                        record.setdefault("extra_embeddings", []).append(feature)
                        changed = True

                if (legacy_label == "boss" and not had_current_memory
                        and not any(person["role"] == "boss"
                                    for person in known_people.values())):
                    known_people[label]["role"] = "boss"
                    changed = True
    except Exception as exc:
        print(f"WARNING: Could not import legacy face memory: {exc}")
        return

    if changed:
        save_memory()


load_memory()
import_legacy_memory()


# ============================================================
# TRACKING STATE
# ============================================================

# BoT-SORT track ID -> persistent identity label.
# Once a track is recognized, its identity stays attached even
# on frames where the face is not visible.
track_identity = {}

# For new faces we collect several embeddings before auto-enrollment.
pending_enrollment = {}

# The selected TARGET identity is restored from persistent memory.
target_track_id = None
target_label = next(
    (label for label, record in known_people.items() if record["role"] == "target"),
    None,
)

# Selection only chooses who the next T or B key applies to.
selected_track_id = None
selected_label = None
pending_click = None
pending_boss_track_id = None

palm_hold = PalmHoldDetector(
    PALM_HOLD_SECONDS,
    PALM_MIN_FRAMES,
    PALM_COOLDOWN_SECONDS,
)
last_hand_timestamp_ms = -1
last_hand_processed_timestamp_ms = -1
command_until = 0.0
command_boss_track_id = None
boss_verified_at = {}
boss_last_face = {}
face_cache = {}
face_scan = PeriodicTask(FACE_DETECT_INTERVAL_SECONDS)
hand_submit = PeriodicTask(HAND_SUBMIT_INTERVAL_SECONDS)


def submit_hand_frame(frame, captured_at):
    """Sample hands from the camera thread, independently of YOLO speed."""
    global last_hand_timestamp_ms
    if not hand_submit.due(captured_at):
        return
    height, width = frame.shape[:2]
    hand_frame = frame
    if width > HAND_INPUT_MAX_WIDTH:
        scale = HAND_INPUT_MAX_WIDTH / width
        hand_frame = cv2.resize(
            frame, (HAND_INPUT_MAX_WIDTH, max(1, int(height * scale)))
        )
    hand_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB),
    )
    last_hand_timestamp_ms = max(
        last_hand_timestamp_ms + 1, int(captured_at * 1000)
    )
    hand_landmarker.detect_async(hand_image, last_hand_timestamp_ms)


# ============================================================
# HELPERS
# ============================================================

def point_inside_box(px, py, box):
    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and y1 <= py <= y2
    )


def get_face_feature(frame, face):
    aligned_face = face_recognizer.alignCrop(
        frame,
        face
    )

    feature = face_recognizer.feature(
        aligned_face
    )

    return normalize_feature(
        feature
    )


def get_person_by_track_id(
    people,
    track_id
):
    if track_id is None:
        return None

    for person in people:
        if person["track_id"] == track_id:
            return person

    return None


def remember_person_now(person):
    """Save a selected visible face before assigning a persistent role."""
    if person["identity"] is not None:
        return person["identity"]

    track_id = person["track_id"]
    features = pending_enrollment.get(track_id, [])
    if not features and person["face_feature"] is not None:
        features = [person["face_feature"]]
    if not features:
        return None

    averaged_feature = normalize_feature(
        np.mean(np.concatenate(features, axis=0), axis=0, keepdims=True)
    )
    label, _ = enroll_new_identity(averaged_feature)
    track_identity[track_id] = label
    person["identity"] = label
    pending_enrollment.pop(track_id, None)
    return label


def clear_target():
    global target_track_id
    global target_label

    if get_role(target_label) == "target":
        known_people[target_label]["role"] = "known"
        save_memory()

    target_track_id = None
    target_label = None


def send_arm_command(name, *values, frame_captured_at=None):
    """Send intent and frame age; the ROS node owns all joint decisions."""
    parts = [name, *(f"{value:.4f}" for value in values)]
    if frame_captured_at is not None:
        parts.append(str(max(0, round(
            (time.monotonic() - frame_captured_at) * 1000
        ))))
    try:
        arm_udp_socket.sendto(
            ",".join(parts).encode("ascii"), (ARM_UDP_HOST, ARM_UDP_PORT)
        )
    except OSError as exc:
        print(f"Arm UDP send failed: {exc}")


def emit_boss_palm_command(boss_person, width, height, frame_captured_at):
    """Report the boss's open-palm event and request a bow."""
    face = boss_person["face"]
    face_center = [
        round(float(face[0] + face[2] / 2), 1),
        round(float(face[1] + face[3] / 2), 1),
    ]
    event = {
        "event": "boss_open_palm",
        "boss_identity": boss_person["identity"],
        "boss_track_id": boss_person["track_id"],
        "face_center_px": face_center,
        "frame_size_px": [width, height],
        "frame_age_ms": round((time.monotonic() - frame_captured_at) * 1000),
        "sequence": ["stop", "face_boss", "bow"],
    }
    print(json.dumps(event), flush=True)
    send_arm_command("BOW", frame_captured_at=frame_captured_at)


def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):
    global pending_click

    if event == cv2.EVENT_LBUTTONDOWN:
        pending_click = (x, y)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera {CAMERA_ID}"
    )

camera = LatestFrameCapture(cap, on_frame=submit_hand_frame).start()

cv2.namedWindow(
    WINDOW_NAME
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


print()
print("Controls:")
print("  CLICK PERSON : select them")
print("  t            : toggle TARGET for selected person")
print("  b            : make selected person BOSS")
print("  CLICK EMPTY / ESC : clear selection")
print("  BOSS open palm (0.5 s) : submit command")
print("  q           : quit")
print()


# ============================================================
# MAIN LOOP
# ============================================================

last_camera_sequence = 0
while True:

    ret, frame, last_camera_sequence, frame_captured_at = camera.read_latest(
        last_camera_sequence
    )

    if not ret:
        print(
            "Failed to read camera."
        )
        break

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # YOLO26n + BoT-SORT
    # --------------------------------------------------------

    results = yolo.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0],
        imgsz=YOLO_INPUT_SIZE,
        verbose=False
    )

    people = []

    if (
        results
        and results[0].boxes is not None
        and results[0].boxes.id is not None
    ):

        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            results[0]
            .boxes
            .id
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            results[0]
            .boxes
            .conf
            .cpu()
            .numpy()
        )

        for box, track_id, confidence in zip(
            boxes,
            track_ids,
            confidences
        ):

            x1, y1, x2, y2 = (
                box.astype(int)
            )

            people.append({
                "track_id": track_id,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "confidence": float(
                    confidence
                ),
                "face": None,
                "face_feature": None,
                "identity": (
                    track_identity.get(
                        track_id
                    )
                ),
                "identity_score": None,
            })


    # --------------------------------------------------------
    # YUNET FACE DETECTION
    # --------------------------------------------------------

    face_now = time.monotonic()
    visible_track_ids = {person["track_id"] for person in people}
    for track_id in list(face_cache):
        if track_id not in visible_track_ids:
            del face_cache[track_id]

    for person in people:
        person["face"] = shifted_recent_face(
            face_cache.get(person["track_id"]), person["box"],
            face_now, FACE_CACHE_MAX_AGE_SECONDS,
        )

    if people and face_scan.due(face_now):
        face_detector.setInputSize((width, height))
        _, faces = face_detector.detect(frame)

        if faces is not None:
            for face in faces:
                fx, fy, fw, fh = (int(value) for value in face[:4])
                face_center_x = fx + fw // 2
                face_center_y = fy + fh // 2

                # Associate the face with the smallest containing person box.
                candidates = []
                for person in people:
                    if point_inside_box(face_center_x, face_center_y,
                                        person["box"]):
                        x1, y1, x2, y2 = person["box"]
                        candidates.append((max(1, x2 - x1) * max(1, y2 - y1),
                                           person))
                if not candidates:
                    continue

                _, owner = min(candidates, key=lambda item: item[0])
                owner["face"] = face
                track_id = owner["track_id"]
                face_cache[track_id] = (face.copy(), owner["box"], face_now)

                identity = track_identity.get(track_id)
                needs_feature = needs_face_feature(
                    identity, get_role(identity), boss_verified_at.get(track_id),
                    face_now, BOSS_FACE_VERIFY_INTERVAL_SECONDS,
                )
                if needs_feature:
                    try:
                        owner["face_feature"] = get_face_feature(frame, face)
                    except cv2.error:
                        owner["face_feature"] = None


    # --------------------------------------------------------
    # PERMANENT IDENTITY RECOGNITION / AUTO ENROLLMENT
    # --------------------------------------------------------

    # Drop pending enrollment buffers for tracks that vanished.
    for track_id in list(
        pending_enrollment.keys()
    ):
        if track_id not in visible_track_ids:
            del pending_enrollment[
                track_id
            ]


    for person in people:

        track_id = person["track_id"]
        feature = person["face_feature"]

        # Identity already attached to this BoT-SORT track.
        if track_id in track_identity:

            person["identity"] = (
                track_identity[
                    track_id
                ]
            )

            continue


        if feature is None:
            continue


        # First try the permanent database.
        label, score = recognize_from_memory(
            feature
        )

        if label is not None:

            track_identity[track_id] = (
                label
            )

            person["identity"] = label
            person["identity_score"] = (
                score
            )

            pending_enrollment.pop(
                track_id,
                None
            )

            continue


        # Truly unknown-looking face.
        # Do not immediately create a new identity.
        pending_enrollment.setdefault(
            track_id,
            []
        )

        pending_enrollment[
            track_id
        ].append(
            feature.copy()
        )


        if (
            len(
                pending_enrollment[
                    track_id
                ]
            )
            < AUTO_ENROLL_FRAMES
        ):
            continue


        # Average several frames to make accidental duplicate
        # enrollment much less likely.
        averaged_feature = (
            normalize_feature(
                np.mean(
                    np.concatenate(
                        pending_enrollment[
                            track_id
                        ],
                        axis=0
                    ),
                    axis=0,
                    keepdims=True
                )
            )
        )


        # Check memory AGAIN using the averaged embedding.
        label, score = recognize_from_memory(
            averaged_feature
        )


        if label is None:

            label, _ = enroll_new_identity(
                averaged_feature
            )


        track_identity[track_id] = (
            label
        )

        person["identity"] = label

        pending_enrollment.pop(
            track_id,
            None
        )


    # --------------------------------------------------------
    # PROCESS CLICK SELECTION
    # --------------------------------------------------------

    selected_person = get_person_by_track_id(people, selected_track_id)
    if selected_person is None and selected_label is not None:
        selected_person = next(
            (person for person in people if person["identity"] == selected_label),
            None
        )
        if selected_person is not None:
            selected_track_id = selected_person["track_id"]

    if selected_person is None:
        selected_track_id = None
        selected_label = None
        pending_boss_track_id = None
    else:
        selected_label = selected_person["identity"]

    if pending_click is not None:

        click_x, click_y = pending_click
        pending_click = None
        pending_boss_track_id = None

        clicked_people = []

        for person in people:

            if not point_inside_box(
                click_x,
                click_y,
                person["box"]
            ):
                continue

            x1, y1, x2, y2 = (
                person["box"]
            )

            area = (
                max(1, x2 - x1)
                * max(1, y2 - y1)
            )

            clicked_people.append(
                (area, person)
            )


        if not clicked_people:
            selected_track_id = None
            selected_label = None
        else:
            _, selected_person = min(clicked_people, key=lambda item: item[0])
            selected_track_id = selected_person["track_id"]
            selected_label = selected_person["identity"]

    if pending_boss_track_id is not None:
        pending_boss = get_person_by_track_id(people, pending_boss_track_id)
        if pending_boss is None:
            pending_boss_track_id = None
        elif pending_boss["identity"] is not None:
            make_boss(pending_boss["identity"])
            if (target_label == pending_boss["identity"]
                    or target_track_id == pending_boss_track_id):
                clear_target()
            pending_boss_track_id = None


    # --------------------------------------------------------
    # IF TARGET WAS SELECTED BEFORE IDENTITY ENROLLMENT,
    # ATTACH ITS PERMANENT LABEL ONCE AVAILABLE
    # --------------------------------------------------------

    target_person = (
        get_person_by_track_id(
            people,
            target_track_id
        )
    )

    if (
        target_person is not None
        and target_label is None
        and target_person[
            "identity"
        ] is not None
    ):

        candidate_label = (
            target_person[
                "identity"
            ]
        )

        target_label = candidate_label
        make_target(candidate_label)


    # --------------------------------------------------------
    # TARGET REACQUISITION
    # --------------------------------------------------------

    target_person = (
        get_person_by_track_id(
            people,
            target_track_id
        )
    )

    target_visible = (
        target_person is not None
    )


    if (
        not target_visible
        and target_label is not None
    ):

        # First use remembered identity attached to active body tracks.
        for person in people:

            if (
                person["identity"]
                == target_label
            ):

                old_track_id = (
                    target_track_id
                )

                target_track_id = (
                    person["track_id"]
                )

                target_person = person
                target_visible = True

                print(
                    "TARGET REACQUIRED: "
                    f"{old_track_id} -> "
                    f"{target_track_id} "
                    f"({target_label})"
                )

                break


        # If the new BoT-SORT track has not yet inherited an identity,
        # directly compare visible faces against the target's saved samples.
        if (
            not target_visible
            and target_label
            in known_people
        ):

            target_record = known_people[target_label]
            target_samples = [
                target_record["embedding"],
                *target_record.get("extra_embeddings", []),
            ]

            best_person = None
            best_score = -1.0

            for person in people:

                feature = (
                    person[
                        "face_feature"
                    ]
                )

                if feature is None:
                    continue

                score = max(
                    compare_faces(sample, feature)
                    for sample in target_samples
                )

                if score > best_score:
                    best_score = score
                    best_person = person


            if (
                best_person is not None
                and best_score
                >= FACE_MATCH_THRESHOLD
            ):

                old_track_id = (
                    target_track_id
                )

                target_track_id = (
                    best_person[
                        "track_id"
                    ]
                )

                track_identity[
                    target_track_id
                ] = target_label

                best_person[
                    "identity"
                ] = target_label

                target_person = (
                    best_person
                )

                target_visible = True

                print(
                    "TARGET FACE REACQUIRED: "
                    f"{old_track_id} -> "
                    f"{target_track_id} "
                    f"(score "
                    f"{best_score:.3f})"
                )


    # --------------------------------------------------------
    # BOSS/TARGET MUTUAL EXCLUSION
    # --------------------------------------------------------

    if (
        target_label is not None
        and get_role(
            target_label
        ) == "boss"
    ):

        clear_target()

        target_person = None
        target_visible = False


    # --------------------------------------------------------
    # BOSS OPEN-PALM COMMAND
    # --------------------------------------------------------

    now = time.monotonic()
    with hand_output_lock:
        hand_result, hand_result_timestamp_ms = matching_hand_result(
            hand_outputs, frame_captured_at * 1000, HAND_RESULT_MAX_AGE_MS,
        )

    visible_boss_track_ids = set()
    for person in people:
        if get_role(person["identity"]) != "boss":
            continue
        track_id = person["track_id"]
        visible_boss_track_ids.add(track_id)
        if person["face_feature"] is not None:
            verified_label, _ = recognize_from_memory(person["face_feature"])
            if verified_label == person["identity"]:
                boss_verified_at[track_id] = now
                boss_last_face[track_id] = person["face"].copy()

    for track_id, verified_at in list(boss_verified_at.items()):
        if (track_id not in visible_boss_track_ids
                or now - verified_at > BOSS_FACE_MAX_AGE_SECONDS):
            del boss_verified_at[track_id]
            boss_last_face.pop(track_id, None)

    for person in people:
        if person["track_id"] not in boss_verified_at:
            continue
        face = person["face"]
        if face is not None:
            boss_x = float(face[0] + face[2] / 2)
            boss_y = float(face[1] + face[3] / 2)
        else:
            x1, y1, x2, y2 = person["box"]
            boss_x = (x1 + x2) / 2
            boss_y = y1 + 0.18 * (y2 - y1)
        send_arm_command(
            "BOSS_PRESENT",
            max(-1.0, min(1.0, (boss_x - width / 2) / (width / 2))),
            max(-1.0, min(1.0, (boss_y - height / 2) / (height / 2))),
            frame_captured_at=frame_captured_at,
        )
        break

    boss_palm_track_id = None
    boss_hand = None
    hand_status = None
    if hand_result is not None:
        boss_palm_track_id, boss_hand, hand_status = boss_palm_owner(
            hand_result, people, set(boss_verified_at), width, height,
            visible_boss_track_ids,
        )

    if hand_result_timestamp_ms > last_hand_processed_timestamp_ms:
        last_hand_processed_timestamp_ms = hand_result_timestamp_ms
        if palm_hold.update(now, boss_palm_track_id):
            boss_person = get_person_by_track_id(people, boss_palm_track_id)
            if boss_person is not None:
                boss_person = dict(boss_person)
                boss_person["face"] = boss_last_face[boss_palm_track_id]
                emit_boss_palm_command(
                    boss_person, width, height, frame_captured_at
                )
                command_until = now + COMMAND_DISPLAY_SECONDS
                command_boss_track_id = boss_palm_track_id
    elif hand_result is None:
        palm_hold.update(now, None)


    # --------------------------------------------------------
    # TRACKING POINT
    # --------------------------------------------------------

    target_point = None
    if target_person is not None:

        # Prefer actual detected face center.
        if target_person[
            "face"
        ] is not None:

            face = (
                target_person[
                    "face"
                ]
            )

            fx = float(face[0])
            fy = float(face[1])
            fw = float(face[2])
            fh = float(face[3])

            target_x = (
                fx + fw / 2.0
            )

            target_y = (
                fy + fh / 2.0
            )

            target_point = (
                target_x,
                target_y
            )

        # Face temporarily lost, but the body tracker
        # still knows which person is the target.
        elif BODY_FALLBACK:

            x1, y1, x2, y2 = (
                target_person[
                    "box"
                ]
            )

            target_x = (
                x1 + x2
            ) / 2.0

            # Approximate head/upper torso rather than body center.
            target_y = (
                y1
                + 0.18
                * (y2 - y1)
            )

            target_point = (
                target_x,
                target_y
            )

    if now < command_until:
        target_point = None

    if target_point is None:
        shoulder_framing.observe(frame, None, frame_captured_at)
        # Distinguish a still-selected but missing person from clearing the
        # target or suppressing tracking for a boss command. The original arm
        # bridge ignores TARGET_MISSING and stops on its existing image timeout.
        missing_selected = (now >= command_until
                            and (target_label is not None or target_track_id is not None))
        send_arm_command("TARGET_MISSING" if missing_selected else "TARGET_LOST",
                         frame_captured_at=frame_captured_at)
    else:
        send_arm_command(
            "TRACK",
            max(-1.0, min(1.0, (target_point[0] - width / 2) / (width / 2))),
            max(-1.0, min(1.0, (target_point[1] - height / 2) / (height / 2))),
            frame_captured_at=frame_captured_at,
        )
        # Optional FB receiver uses BODY height only. The original arm bridge
        # ignores BOX packets, leaving its existing tracking protocol intact.
        if target_person is not None:
            shoulder = shoulder_framing.observe(frame, target_person, frame_captured_at)
            if shoulder is not None and time.monotonic()-shoulder[3] <= .35:
                size, identity, valid, captured = shoulder
                send_arm_command('SHOULDERS', size, identity, int(valid), frame_captured_at=captured)
            else:
                send_arm_command('SHOULDERS', 0, int(target_person['track_id']), 0,
                                 frame_captured_at=frame_captured_at)
            box_height, box_quality = body_height_signal(
                target_person["box"], target_person.get("confidence", 0.0), width, height)
            send_arm_command(
                "BOX", box_height, int(target_person["track_id"]),
                box_quality, frame_captured_at=frame_captured_at,
            )

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    for person in people:

        x1, y1, x2, y2 = (
            person["box"]
        )

        label = (
            person["identity"]
        )

        role = get_role(
            label
        )

        is_target = (
            target_track_id
            == person[
                "track_id"
            ]
        )
        is_selected = selected_track_id == person["track_id"]


        if role == "boss":

            color = (
                255,
                0,
                255
            )

            thickness = 4

            text = "BOSS"


        elif is_target:

            color = (
                0,
                0,
                255
            )

            thickness = 4

            text = "TARGET"


        else:

            color = (
                0,
                255,
                0
            )

            thickness = 2

            text = None


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        if is_selected:
            cv2.rectangle(
                frame,
                (x1 - 5, y1 - 5),
                (x2 + 5, y2 + 5),
                (255, 255, 255),
                2
            )
            text = f"{text} / PERSON SELECTED" if text else "PERSON SELECTED"


        if text is not None:
            cv2.putText(
                frame,
                text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                color,
                2
            )


        if (
            person["face"]
            is not None
        ):

            face = (
                person["face"]
            )

            fx = int(face[0])
            fy = int(face[1])
            fw = int(face[2])
            fh = int(face[3])

            cv2.rectangle(
                frame,
                (fx, fy),
                (
                    fx + fw,
                    fy + fh
                ),
                (
                    255,
                    255,
                    0
                ),
                2
            )

    if hand_result is not None:
        for x1, y1, x2, y2 in hand_boxes(hand_result, width, height):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, "HAND", (x1, max(25, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    if boss_hand is not None:
        hand_points = [
            (int(point.x * width), int(point.y * height))
            for point in boss_hand
        ]
        for finger in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8),
                       (0, 9, 10, 11, 12), (0, 13, 14, 15, 16),
                       (0, 17, 18, 19, 20)):
            for start, end in zip(finger, finger[1:]):
                cv2.line(frame, hand_points[start], hand_points[end],
                         (0, 255, 255), 2)
        cv2.putText(
            frame, "BOSS PALM", hand_points[0],
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )


    # Camera center.
    camera_center = (
        width // 2,
        height // 2
    )


    cv2.drawMarker(
        frame,
        camera_center,
        (
            255,
            255,
            255
        ),
        cv2.MARKER_CROSS,
        28,
        2
    )


    if target_point is not None:

        tx = int(
            target_point[0]
        )

        ty = int(
            target_point[1]
        )

        cv2.circle(
            frame,
            (tx, ty),
            7,
            (
                0,
                0,
                255
            ),
            -1
        )

        cv2.line(
            frame,
            camera_center,
            (tx, ty),
            (
                0,
                0,
                255
            ),
            2
        )

    if now < command_until:
        command_boss = get_person_by_track_id(people, command_boss_track_id)
        if command_boss is not None and command_boss["face"] is not None:
            face = command_boss["face"]
            boss_point = (
                int(face[0] + face[2] / 2),
                int(face[1] + face[3] / 2),
            )
            cv2.line(frame, camera_center, boss_point, (255, 0, 255), 2)
            cv2.circle(frame, boss_point, 8, (255, 0, 255), -1)

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if now < command_until:
        status = "BOSS COMMAND: STOP > FACE BOSS > BOW"
    elif boss_palm_track_id is not None and palm_hold.latched:
        status = "BOSS PALM - RELEASE TO REARM"
    elif boss_palm_track_id is not None:
        held = min(PALM_HOLD_SECONDS, now - palm_hold.first_seen)
        status = f"BOSS PALM - HOLD {held:.1f}/{PALM_HOLD_SECONDS:.1f}s"
    elif hand_result is not None and hand_result.hand_landmarks:
        status = hand_status
    elif pending_boss_track_id is not None:
        status = "BOSS: WAITING FOR FACE"
    elif target_track_id is not None and target_label is None:
        status = "TARGET: SHOW FACE TO SAVE"
    elif selected_track_id is not None:
        status = "PERSON SELECTED - T: TARGET  B: BOSS"
    elif target_label is None:
        status = "NO TARGET"
    elif target_visible:
        status = "TARGET TRACKING"
    else:
        status = "TARGET LOST"


    status_scale = 0.66
    text_width, _ = cv2.getTextSize(
        status, cv2.FONT_HERSHEY_SIMPLEX, status_scale, 2
    )[0]
    if text_width > width - 40:
        status_scale *= (width - 40) / text_width
    status_color = (
        (0, 255, 0) if now < command_until
        else (0, 255, 255) if hand_result is not None and hand_result.hand_landmarks
        else (255, 255, 255)
    )
    cv2.rectangle(frame, (10, 8), (width - 10, 46), (20, 20, 20), -1)
    cv2.putText(
        frame,
        status,
        (
            20,
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        status_scale,
        status_color,
        2
    )

    cv2.putText(
        frame, "HAND VISION ON", (20, height - 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
    )


    cv2.putText(
        frame,
        (
            "Click: select | "
            "T: target | "
            "B: boss | "
            "Empty/Esc: deselect | "
            "Q: quit"
        ),
        (
            20,
            height - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (
            255,
            255,
            255
        ),
        1
    )


    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    if key in (ord("q"), ord("Q")):
        break


    if key == 27:
        selected_track_id = None
        selected_label = None
        pending_boss_track_id = None

    if key in (ord("t"), ord("T")) and selected_track_id is not None:
        pending_boss_track_id = None
        selected_person = get_person_by_track_id(people, selected_track_id)
        if selected_person is not None:
            label = remember_person_now(selected_person)
            selected_label = label
            if (target_track_id == selected_track_id
                    or (label is not None and target_label == label)):
                clear_target()
            else:
                if label is None:
                    clear_target()
                else:
                    make_target(label)
                target_track_id = selected_track_id
                target_label = label

    if key in (ord("b"), ord("B")) and selected_track_id is not None:
        selected_person = get_person_by_track_id(people, selected_track_id)
        if selected_person is not None:
            label = remember_person_now(selected_person)
            selected_label = label
            if label is None:
                pending_boss_track_id = selected_track_id
            else:
                pending_boss_track_id = None
                if get_role(label) != "boss":
                    make_boss(label)
                if target_label == label or target_track_id == selected_track_id:
                    clear_target()


camera.close()
shoulder_framing.close()
hand_landmarker.close()
send_arm_command("STOP")
arm_udp_socket.close()
cv2.destroyAllWindows()
