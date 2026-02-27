import cv2
import numpy as np
from ultralytics import YOLO

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
            S = P_pred @ self.H.T
            K = S @ np.linalg.inv(self.H @ S + self.R)
            self.S = S_pred + K @ y
            self.P = (self.I - K @ self.H) @ P_pred
        else:
            self.S = S_pred
            self.P = P_pred

        return self.S[0], self.S[3]

cap = cv2.VideoCapture("VİDEO_YOLU.mp4")
model = YOLO("MODEL_YOLU.pt", verbose=False)

tracker = None
kalman_box_size = 80
lost_counter = 0
max_lost_frames = 60

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    tespit = None

    av_x_min, av_x_max = int(w * 0.25), int(w * 0.75)
    av_y_min, av_y_max = int(h * 0.10), int(h * 0.90)
    av_cx, av_cy = (av_x_min + av_x_max) // 2, (av_y_min + av_y_max) // 2
    cv2.rectangle(frame, (av_x_min, av_y_min), (av_x_max, av_y_max), (0, 255, 255), 2)  # Sarı

    if tracker is None:
        results = model(frame, verbose=False)[0]
        if len(results.boxes.xyxy) > 0:
            x1, y1, x2, y2 = results.boxes.xyxy[0].cpu().numpy()
            tespit = [(x1 + x2) / 2, (y1 + y2) / 2]
            tracker = SimpleKalman(x=tespit[0], y=tespit[1])
            lost_counter = 0
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)  # Yeşil
            cv2.line(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), (av_cx, av_cy), (0, 0, 0), 2)
    else:
        x_kalman, y_kalman = tracker.step()

        x1_crop = int(x_kalman - kalman_box_size // 2)
        y1_crop = int(y_kalman - kalman_box_size // 2)
        x2_crop = int(x_kalman + kalman_box_size // 2)
        y2_crop = int(y_kalman + kalman_box_size // 2)

        x1_crop = max(0, x1_crop)
        y1_crop = max(0, y1_crop)
        x2_crop = min(frame.shape[1], x2_crop)
        y2_crop = min(frame.shape[0], y2_crop)

        crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if crop.size == 0:
            lost_counter += 1
            continue

        results = model(crop, verbose=False)[0]

        if len(results.boxes.xyxy) > 0:
            bx1, by1, bx2, by2 = results.boxes.xyxy[0].cpu().numpy()
            cx = x1_crop + (bx1 + bx2) / 2
            cy = y1_crop + (by1 + by2) / 2
            tespit = [cx, cy]
            lost_counter = 0

            cv2.rectangle(frame, (int(cx - 20), int(cy - 20)), (int(cx + 20), int(cy + 20)), (0,0,255), 2)
            cv2.line(frame, (int(cx), int(cy)), (av_cx, av_cy), (0, 0, 0), 2)
        else:
            lost_counter += 1

        x_kalman, y_kalman = tracker.step(tespit)

        if lost_counter > max_lost_frames:
            tracker = None
            lost_counter = 0

    cv2.imshow("YOLOv8 + Kalman Tracking + AV Çerçeve", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
