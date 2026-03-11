import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from System import (
    EYE_CLOSED_FACTOR,
    HEAD_TILT_THRESHOLD,
    LEFT_EYE,
    MAR,
    RIGHT_EYE,
    WINDOW,
    YAWN_FACTOR,
    EAR,
    get_head_tilt,
    mp_face_mesh,
)


class DrowsinessDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Driver Drowsiness Detection Dashboard")
        self.root.geometry("1260x760")
        self.root.configure(bg="#0B1118")
        self.root.minsize(1100, 680)

        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

        self.ear_history = []
        self.mar_history = []
        self.tilt_history = []

        self.baseline_ear = 0.0
        self.baseline_mar = 0.0
        self.baseline_tilt = 0.0
        self.calibration_frames = 50
        self.frame_count = 0
        self.calibrated = False

        self.last_ear = 0.0
        self.last_mar = 0.0
        self.last_tilt = 0.0
        self.last_alertness = 100.0
        self.last_state = "CALIBRATING"

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")

        self._build_styles()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_frame()

    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Card.TFrame", background="#111A24")
        style.configure("Title.TLabel", background="#0B1118", foreground="#E8EEF6", font=("Segoe UI", 24, "bold"))
        style.configure("Subtitle.TLabel", background="#0B1118", foreground="#8DA3B8", font=("Segoe UI", 11))
        style.configure("CardTitle.TLabel", background="#111A24", foreground="#AFC3D6", font=("Segoe UI", 10, "bold"))
        style.configure("Metric.TLabel", background="#111A24", foreground="#F4F8FC", font=("Segoe UI", 18, "bold"))
        style.configure("Status.TLabel", background="#111A24", foreground="#0B1118", font=("Segoe UI", 14, "bold"), padding=8)

    def _build_layout(self):
        top = ttk.Frame(self.root, style="Card.TFrame")
        top.pack(fill="x", padx=18, pady=(16, 10))

        title_block = ttk.Frame(top, style="Card.TFrame")
        title_block.pack(fill="x", padx=14, pady=14)

        ttk.Label(title_block, text="Driver Drowsiness Monitor", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_block,
            text="Computer Vision + Mediapipe Face Mesh | Final Year Project Demo",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(self.root, style="Card.TFrame")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 16))
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        video_card = ttk.Frame(body, style="Card.TFrame")
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.video_label = tk.Label(
            video_card,
            bg="#05080D",
            fg="#D3E2F1",
            text="Initializing camera...",
            font=("Segoe UI", 14, "bold"),
        )
        self.video_label.pack(fill="both", expand=True, padx=12, pady=12)

        side = ttk.Frame(body, style="Card.TFrame")
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)

        status_card = ttk.Frame(side, style="Card.TFrame")
        status_card.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        ttk.Label(status_card, text="Current State", style="CardTitle.TLabel").pack(anchor="w", padx=12, pady=(10, 6))
        self.state_label = tk.Label(
            status_card,
            text="CALIBRATING",
            bg="#F7C948",
            fg="#111111",
            font=("Segoe UI", 16, "bold"),
            padx=14,
            pady=8,
        )
        self.state_label.pack(anchor="w", padx=12, pady=(0, 10))

        alert_card = ttk.Frame(side, style="Card.TFrame")
        alert_card.grid(row=1, column=0, sticky="ew", padx=10, pady=8)
        ttk.Label(alert_card, text="Alertness", style="CardTitle.TLabel").pack(anchor="w", padx=12, pady=(10, 4))
        self.alert_value = ttk.Label(alert_card, text="100%", style="Metric.TLabel")
        self.alert_value.pack(anchor="w", padx=12)
        self.alert_bar = ttk.Progressbar(alert_card, orient="horizontal", mode="determinate", length=250, maximum=100)
        self.alert_bar.pack(fill="x", padx=12, pady=(8, 12))

        metrics_card = ttk.Frame(side, style="Card.TFrame")
        metrics_card.grid(row=2, column=0, sticky="ew", padx=10, pady=8)
        ttk.Label(metrics_card, text="Live Metrics", style="CardTitle.TLabel").pack(anchor="w", padx=12, pady=(10, 6))

        self.ear_label = ttk.Label(metrics_card, text="EAR: 0.00", style="Metric.TLabel")
        self.ear_label.pack(anchor="w", padx=12, pady=2)
        self.mar_label = ttk.Label(metrics_card, text="MAR: 0.00", style="Metric.TLabel")
        self.mar_label.pack(anchor="w", padx=12, pady=2)
        self.tilt_label = ttk.Label(metrics_card, text="Tilt: 0", style="Metric.TLabel")
        self.tilt_label.pack(anchor="w", padx=12, pady=(2, 10))

        info_card = ttk.Frame(side, style="Card.TFrame")
        info_card.grid(row=3, column=0, sticky="ew", padx=10, pady=(8, 10))
        self.info_label = ttk.Label(
            info_card,
            text="Calibrating baseline... Keep your face steady.",
            style="Subtitle.TLabel",
            wraplength=300,
            justify="left",
        )
        self.info_label.pack(anchor="w", padx=12, pady=12)

    def _set_state_ui(self, state):
        if state == "ALERT":
            bg, fg = "#1FDE8A", "#072B1A"
        elif state == "DROWSY":
            bg, fg = "#FFB74D", "#3D2102"
        elif state == "SLEEPY":
            bg, fg = "#FF6B6B", "#3C0909"
        else:
            bg, fg = "#F7C948", "#111111"
        self.state_label.configure(text=state, bg=bg, fg=fg)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.info_label.configure(text="Camera read failed.")
            self.root.after(30, self.update_frame)
            return

        frame = cv2.resize(frame, (960, 600))
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        state = "CALIBRATING" if not self.calibrated else self.last_state
        status_text = "Calibrating baseline... Keep your face steady."

        if results.multi_face_landmarks:
            landmarks = np.array(
                [[lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark]
            )

            left_ear = EAR(landmarks, LEFT_EYE)
            right_ear = EAR(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2
            mar = MAR(landmarks)
            raw_tilt = get_head_tilt(landmarks)

            self.last_ear = avg_ear
            self.last_mar = mar

            if not self.calibrated:
                self.baseline_ear += avg_ear
                self.baseline_mar += mar
                self.baseline_tilt += raw_tilt
                self.frame_count += 1
                progress = int((self.frame_count / self.calibration_frames) * 100)
                status_text = f"Calibrating... {min(progress, 100)}%"

                if self.frame_count >= self.calibration_frames:
                    self.baseline_ear /= self.calibration_frames
                    self.baseline_mar /= self.calibration_frames
                    self.baseline_tilt /= self.calibration_frames
                    self.calibrated = True
                    status_text = "Calibration complete. Monitoring in real time."
            else:
                tilt = abs(raw_tilt - self.baseline_tilt)
                self.last_tilt = tilt

                self.ear_history.append(avg_ear)
                self.mar_history.append(mar)
                self.tilt_history.append(tilt)

                if len(self.ear_history) > WINDOW:
                    self.ear_history.pop(0)
                    self.mar_history.pop(0)
                    self.tilt_history.pop(0)

                eye_closed_ratio = (
                    sum(
                        1
                        for ear_v in self.ear_history
                        if ear_v < (self.baseline_ear * EYE_CLOSED_FACTOR)
                    )
                    / len(self.ear_history)
                )
                yawn_ratio = (
                    sum(
                        1
                        for mar_v in self.mar_history
                        if mar_v > (self.baseline_mar * YAWN_FACTOR)
                    )
                    / len(self.mar_history)
                )
                head_down_ratio = (
                    sum(
                        1
                        for tilt_v in self.tilt_history
                        if tilt_v > HEAD_TILT_THRESHOLD
                    )
                    / len(self.tilt_history)
                )

                fatigue_score = min(
                    100 * ((eye_closed_ratio * 0.50) + (yawn_ratio * 0.30) + (head_down_ratio * 0.20)),
                    100,
                )
                alertness = max(0, 100 - fatigue_score)
                self.last_alertness = alertness

                if fatigue_score < 20:
                    state = "ALERT"
                    status_text = "Driver is attentive."
                elif fatigue_score < 50:
                    state = "DROWSY"
                    status_text = "Early signs of fatigue detected."
                else:
                    state = "SLEEPY"
                    status_text = "High fatigue risk. Immediate break advised."

                self.last_state = state
        else:
            status_text = "No face detected. Align your face with the camera."
            state = "NO FACE" if self.calibrated else "CALIBRATING"

        self._set_state_ui(state)
        self.alert_bar["value"] = self.last_alertness
        self.alert_value.configure(text=f"{int(self.last_alertness)}%")
        self.ear_label.configure(text=f"EAR: {self.last_ear:.2f}")
        self.mar_label.configure(text=f"MAR: {self.last_mar:.2f}")
        self.tilt_label.configure(text=f"Tilt: {int(self.last_tilt)}")
        self.info_label.configure(text=status_text)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

        self.root.after(15, self.update_frame)

    def on_close(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


def main():
    root = tk.Tk()
    DrowsinessDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
