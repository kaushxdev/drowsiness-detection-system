import cv2
import mediapipe as mp
import numpy as np

WINDOW = 20
EYE_CLOSED_FACTOR = 0.8
YAWN_FACTOR = 1.4
HEAD_TILT_THRESHOLD = 20


def draw_ui(frame, ear, mar, tilt, alertness, state):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 260), (30, 30, 30), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(
        frame,
        "Driver Monitoring System",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"EAR: {ear:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"MAR: {mar:.2f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Tilt: {int(tilt)}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    bar_length = int(alertness * 2)

    if state == "ALERT":
        color = (0, 255, 0)
    elif state == "DROWSY":
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame, (20, 180), (220, 200), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 180), (20 + bar_length, 200), color, -1)

    cv2.putText(
        frame,
        f"Alertness: {int(alertness)}%",
        (20, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    cv2.putText(
        frame, f"State: {state}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )


mp_face_mesh = mp.solutions.face_mesh


def distance(a, b):
    return np.linalg.norm(a - b)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def EAR(landmarks, eye):
    pts = np.array([landmarks[i] for i in eye])
    return (distance(pts[1], pts[5]) + distance(pts[2], pts[4])) / (
        2.0 * distance(pts[0], pts[3])
    )


def MAR(landmarks):
    vertical = distance(landmarks[13], landmarks[14])
    horizontal = distance(landmarks[78], landmarks[308])
    return vertical / horizontal


def get_head_tilt(landmarks):
    nose = landmarks[1]
    chin = landmarks[152]
    return nose[1] - chin[1]


def main():
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    ear_history = []
    mar_history = []
    tilt_history = []

    baseline_ear = 0
    baseline_mar = 0
    baseline_tilt = 0

    calibrated = False
    calibration_frames = 50
    frame_count = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if results.multi_face_landmarks:
                landmarks = np.array(
                    [[lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark]
                )

                left_ear = EAR(landmarks, LEFT_EYE)
                right_ear = EAR(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2
                mar = MAR(landmarks)
                raw_tilt = get_head_tilt(landmarks)

                if not calibrated:
                    # Collect baseline values during calibration.
                    baseline_ear += avg_ear
                    baseline_mar += mar
                    baseline_tilt += raw_tilt
                    frame_count += 1

                    cv2.putText(
                        frame,
                        "Calibrating...",
                        (200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                    if frame_count >= calibration_frames:
                        baseline_ear /= calibration_frames
                        baseline_mar /= calibration_frames
                        baseline_tilt /= calibration_frames
                        calibrated = True
                else:
                    tilt = abs(raw_tilt - baseline_tilt)
                    ear_history.append(avg_ear)
                    mar_history.append(mar)
                    tilt_history.append(tilt)

                    if len(ear_history) > WINDOW:
                        ear_history.pop(0)
                        mar_history.pop(0)
                        tilt_history.pop(0)

                    smooth_ear = sum(ear_history) / len(ear_history)
                    smooth_mar = sum(mar_history) / len(mar_history)
                    smooth_tilt = sum(tilt_history) / len(tilt_history)

                    eye_closed_ratio = (
                        sum(1 for ear_v in ear_history if ear_v < (baseline_ear * EYE_CLOSED_FACTOR))
                        / len(ear_history)
                    )
                    yawn_ratio = (
                        sum(1 for mar_v in mar_history if mar_v > (baseline_mar * YAWN_FACTOR))
                        / len(mar_history)
                    )
                    head_down_ratio = (
                        sum(1 for tilt_v in tilt_history if tilt_v > HEAD_TILT_THRESHOLD)
                        / len(tilt_history)
                    )

                    fatigue_score = min(
                        100
                        * (
                            (eye_closed_ratio * 0.50)
                            + (yawn_ratio * 0.30)
                            + (head_down_ratio * 0.20)
                        ),
                        100,
                    )
                    alertness = max(0, 100 - fatigue_score)

                    if fatigue_score < 20:
                        state = "ALERT"
                    elif fatigue_score < 50:
                        state = "DROWSY"
                    else:
                        state = "SLEEPY"

                    draw_ui(frame, avg_ear, mar, tilt, alertness, state)
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Driver Monitoring System", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
