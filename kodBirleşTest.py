import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

# --- Simple Kalman tracker ---
class SimpleKalman:
    def __init__(self, x=0, y=0, dt=1/30, std_x=0.001, std_y=0.001):
        self.dt = dt
        self.S = np.array([x, 0, 0, y, 0, 0])
        self.F = np.array([
            [1, dt, 0.5*dt**2, 0, 0, 0],
            [0, 1, dt,         0, 0, 0],
            [0, 0, 1,          0, 0, 0],
            [0, 0, 0,          1, dt, 0.5*dt**2],
            [0, 0, 0,          0, 1, dt],
            [0, 0, 0,          0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        self.R = np.array([
            [std_x**2, 0],
            [0, std_y**2]
        ])
        self.Q = np.diag([0.01, 0.05, 0.1, 0.01, 0.05, 0.1])
        self.P = np.diag([2, 2, 10, 10, 10, 10])
        self.I = np.eye(6)

    def step(self, z=None):
        S_pred = self.F @ self.S
        P_pred = self.F @ self.P @ self.F.T + self.Q

        if z is not None:
            y = np.array(z) - self.H @ S_pred
            S_mat = P_pred @ self.H.T
            K = S_mat @ np.linalg.inv(self.H @ S_mat + self.R)
            self.S = S_pred + K @ y
            self.P = (self.I - K @ self.H) @ P_pred
        else:
            self.S = S_pred
            self.P = P_pred

        return self.S[0], self.S[3]

# --- Enhancement for drone frames ---
def enhance_drone(img, boost=2.5):
    blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=0)
    sharp = cv2.addWeighted(img, boost, blurred, 1.0 - boost, 0)
    sharp = cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)
    return sharp


video_path = "C:/Users/tunahan/Desktop/modelTest/dog2.mp4"
model_path = "C:/Users/tunahan/Desktop/modelTest/bestv8.pt"

cap = cv2.VideoCapture(video_path)
model = YOLO(model_path, verbose=False).to("cuda")

tracker = None
kalman_box_size = 160
lost_counter = 0
max_lost_frames = 10
prev_time = 0

# Create windows
cv2.namedWindow("Detection only (YOLOv8)")
cv2.namedWindow("YOLOv8 + Kalman Tracking + AV Çerçeve")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Compute FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # --- Detection-only view ---
    frame_det = frame.copy()
    results_det = model(frame_det, verbose=False)[0]
    for box, conf in zip(results_det.boxes.xyxy.cpu().numpy(), results_det.boxes.conf.cpu().numpy()):
        if conf > 0.3:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame_det, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame_det, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # --- Kalman-tracking view ---
    frame_trk = frame.copy()
    cv2.putText(frame_trk, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    av_x_min, av_x_max = int(w * 0.25), int(w * 0.75)
    av_y_min, av_y_max = int(h * 0.10), int(h * 0.90)
    av_cx, av_cy = (av_x_min + av_x_max) // 2, (av_y_min + av_y_max) // 2
    cv2.rectangle(frame_trk, (av_x_min, av_y_min), (av_x_max, av_y_max), (0, 255, 255), 2)

    tespit = None
    # Initialize tracker if needed
    if tracker is None:
        results = model(frame_trk, verbose=False)[0]
        if len(results.boxes.xyxy) > 0:
            x1, y1, x2, y2 = results.boxes.xyxy[0].cpu().numpy()
            tespit = [(x1 + x2) / 2, (y1 + y2) / 2]
            tracker = SimpleKalman(x=tespit[0], y=tespit[1])
            lost_counter = 0
            cv2.rectangle(frame_trk, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            cv2.line(frame_trk, (int((x1 + x2) / 2), int((y1 + y2) / 2)), (av_cx, av_cy), (0,0,0), 2)
    else:
        x_kalman, y_kalman = tracker.step()
        # Clamp prediction
        x_kalman = np.clip(x_kalman, 0, w-1)
        y_kalman = np.clip(y_kalman, 0, h-1)

        half = kalman_box_size // 2
        x1c = int(max(x_kalman - half, 0))
        y1c = int(max(y_kalman - half, 0))
        x2c = int(min(x_kalman + half, w-1))
        y2c = int(min(y_kalman + half, h-1))

        crop = frame_trk[y1c:y2c, x1c:x2c]
        if crop.size > 0:
            crop_enh = enhance_drone(crop)
            results = model(crop_enh, verbose=False)[0]
            if len(results.boxes.xyxy) > 0:
                bx1, by1, bx2, by2 = results.boxes.xyxy[0].cpu().numpy()
                cx = x1c + (bx1 + bx2)/2
                cy = y1c + (by1 + by2)/2
                tespit = [cx, cy]
                lost_counter = 0
                # Safe draw box
                bh = 20
                x1b = int(max(cx - bh, 0))
                y1b = int(max(cy - bh, 0))
                x2b = int(min(cx + bh, w-1))
                y2b = int(min(cy + bh, h-1))
                if x2b > x1b and y2b > y1b:
                    cv2.rectangle(frame_trk, (x1b, y1b), (x2b, y2b), (0,0,255), 2)
                cv2.line(frame_trk, (int(cx), int(cy)), (av_cx, av_cy), (0,0,0), 2)
            else:
                lost_counter += 1
        else:
            lost_counter += 1

        x_kalman, y_kalman = tracker.step(tespit)
        x_kalman = np.clip(x_kalman, 0, w-1)
        y_kalman = np.clip(y_kalman, 0, h-1)
        cv2.circle(frame_trk, (int(x_kalman), int(y_kalman)), 5, (255, 0, 0), -1)

        if lost_counter > max_lost_frames:
            tracker = None
            lost_counter = 0

    # Show both windows
    cv2.imshow("Detection only (YOLOv8)", frame_det)
    cv2.imshow("YOLOv8 + Kalman Tracking + AV Çerçeve", frame_trk)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
