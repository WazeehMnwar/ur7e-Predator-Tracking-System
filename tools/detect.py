import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

YOLO_MODEL = "yolo26n.pt"

YUNET_MODEL = "face_models/face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_models/face_recognition_sface_2021dec.onnx"

CAMERA_ID = 0

# Official OpenCV SFace example uses 0.363 for cosine matching.
FACE_MATCH_THRESHOLD = 0.363


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Loading face detector...")
face_detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.9,      # detection confidence
    0.3,      # NMS
    5000
)

print("Loading face recognizer...")
face_recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)


# ============================================================
# TRACK STATE
# ============================================================

# BoT-SORT track ID -> our simple display number.
display_numbers = {}
next_display_number = 1

# Selected target information.
target_display_number = None
target_track_id = None

# Stored SFace feature vector for selected person.
target_face_feature = None

# If user selected someone while their face wasn't visible,
# capture their face as soon as it becomes available.
waiting_for_target_face = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_display_number(track_id):
    global next_display_number

    if track_id not in display_numbers:
        display_numbers[track_id] = next_display_number
        next_display_number += 1

    return display_numbers[track_id]


def point_inside_box(px, py, box):
    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2 and
        y1 <= py <= y2
    )


def get_face_feature(frame, face):
    """
    Align detected face and calculate SFace embedding.
    """

    aligned_face = face_recognizer.alignCrop(
        frame,
        face
    )

    feature = face_recognizer.feature(
        aligned_face
    )

    return feature


def compare_faces(feature1, feature2):
    """
    Returns cosine similarity.
    Higher = more similar.
    """

    score = face_recognizer.match(
        feature1,
        feature2,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    return float(score)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera {CAMERA_ID}"
    )


print()
print("Controls:")
print("  1-9 : select person")
print("  0   : clear target")
print("  q   : quit")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera.")
        break

    height, width = frame.shape[:2]


    # --------------------------------------------------------
    # YOLO + BoT-SORT
    # --------------------------------------------------------

    results = yolo.track(
        frame,
        persist=True,
        tracker="botsort.yaml",

        # COCO class 0 = person.
        # Ignore everything else.
        classes=[0],

        verbose=False
    )


    people = []


    if (
        results and
        results[0].boxes is not None and
        results[0].boxes.id is not None
    ):

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().numpy()


        for box, track_id, confidence in zip(
            boxes,
            track_ids,
            confidences
        ):

            x1, y1, x2, y2 = box.astype(int)

            display_number = get_display_number(
                track_id
            )

            people.append({
                "track_id": track_id,
                "number": display_number,
                "box": (x1, y1, x2, y2),
                "confidence": float(confidence),
                "face": None,
                "face_feature": None
            })


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    face_detector.setInputSize(
        (width, height)
    )

    _, faces = face_detector.detect(frame)


    if faces is not None:

        for face in faces:

            fx = int(face[0])
            fy = int(face[1])
            fw = int(face[2])
            fh = int(face[3])

            face_center_x = fx + fw // 2
            face_center_y = fy + fh // 2


            # Associate this face with whichever YOLO
            # person bounding box contains its center.
            for person in people:

                if point_inside_box(
                    face_center_x,
                    face_center_y,
                    person["box"]
                ):

                    person["face"] = face

                    try:

                        person["face_feature"] = (
                            get_face_feature(
                                frame,
                                face
                            )
                        )

                    except cv2.error:
                        person["face_feature"] = None

                    break


    # --------------------------------------------------------
    # TARGET FACE ENROLLMENT
    # --------------------------------------------------------

    if (
        target_track_id is not None and
        target_face_feature is None
    ):

        for person in people:

            if (
                person["track_id"] == target_track_id and
                person["face_feature"] is not None
            ):

                target_face_feature = (
                    person["face_feature"].copy()
                )

                waiting_for_target_face = False

                print(
                    f"Face stored for target "
                    f"#{target_display_number}"
                )

                break


    # --------------------------------------------------------
    # FACE RE-IDENTIFICATION
    # --------------------------------------------------------

    target_currently_visible = False


    if target_track_id is not None:

        for person in people:

            if person["track_id"] == target_track_id:
                target_currently_visible = True
                break


    # BoT-SORT lost the original body track.
    # Try to find the stored face among currently visible people.
    if (
        target_display_number is not None and
        target_face_feature is not None and
        not target_currently_visible
    ):

        best_match = None
        best_score = -1


        for person in people:

            feature = person["face_feature"]

            if feature is None:
                continue


            score = compare_faces(
                target_face_feature,
                feature
            )


            if score > best_score:

                best_score = score
                best_match = person


        if (
            best_match is not None and
            best_score >= FACE_MATCH_THRESHOLD
        ):

            old_track = target_track_id

            target_track_id = (
                best_match["track_id"]
            )


            # Preserve the target's original displayed number.
            display_numbers[target_track_id] = (
                target_display_number
            )


            print(
                f"TARGET RE-IDENTIFIED: "
                f"old track {old_track} -> "
                f"new track {target_track_id} "
                f"(face score {best_score:.3f})"
            )


    # --------------------------------------------------------
    # DRAW PEOPLE
    # --------------------------------------------------------

    for person in people:

        x1, y1, x2, y2 = person["box"]

        track_id = person["track_id"]
        number = person["number"]


        is_target = (
            target_track_id == track_id
        )


        if is_target:

            color = (0, 0, 255)
            thickness = 4
            label = f"TARGET #{target_display_number}"

        else:

            color = (0, 255, 0)
            thickness = 2
            label = f"#{number}"


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )


        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


        # Draw face box if detected
        if person["face"] is not None:

            face = person["face"]

            fx = int(face[0])
            fy = int(face[1])
            fw = int(face[2])
            fh = int(face[3])

            cv2.rectangle(
                frame,
                (fx, fy),
                (fx + fw, fy + fh),
                (255, 255, 0),
                2
            )


    # --------------------------------------------------------
    # STATUS TEXT
    # --------------------------------------------------------

    if target_display_number is None:

        status = "TARGET: NONE - press 1-9"

    elif target_face_feature is None:

        status = (
            f"TARGET #{target_display_number} "
            f"- waiting for visible face"
        )

    elif target_currently_visible:

        status = (
            f"TARGET #{target_display_number} "
            f"- TRACKING + FACE STORED"
        )

    else:

        status = (
            f"TARGET #{target_display_number} "
            f"- BODY LOST / SEARCHING FACE"
        )


    cv2.putText(
        frame,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Predator Target Tracking",
        frame
    )


    # --------------------------------------------------------
    # KEYBOARD INPUT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


    # Clear target
    if key == ord("0"):

        print("Target cleared.")

        target_display_number = None
        target_track_id = None
        target_face_feature = None
        waiting_for_target_face = False


    # Select #1 through #9
    if ord("1") <= key <= ord("9"):

        requested_number = int(
            chr(key)
        )


        selected_person = None


        for person in people:

            if person["number"] == requested_number:

                selected_person = person
                break


        if selected_person is None:

            print(
                f"Person #{requested_number} "
                f"is not currently visible."
            )

        else:

            target_display_number = (
                requested_number
            )

            target_track_id = (
                selected_person["track_id"]
            )

            target_face_feature = None


            if (
                selected_person["face_feature"]
                is not None
            ):

                target_face_feature = (
                    selected_person[
                        "face_feature"
                    ].copy()
                )

                waiting_for_target_face = False

                print(
                    f"Selected #{requested_number}. "
                    f"Face stored immediately."
                )

            else:

                waiting_for_target_face = True

                print(
                    f"Selected #{requested_number}. "
                    f"Waiting until their face "
                    f"is visible."
                )


cap.release()
cv2.destroyAllWindows()
