import cv2
import numpy as np
from ultralytics import YOLO
import time

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
        self.Q = np.diag([0.01, 0.02, 0.05, 0.01, 0.02, 0.05])
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


def enhance_drone(img, boost=2.5):
    blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=0)
    sharp = cv2.addWeighted(img, boost, blurred, 1.0 - boost, 0)
    sharp = cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)
    return sharp

# Video ve model yolları
cap = cv2.VideoCapture("C:/Users/tunahan/Desktop/modelTest/dog2.mp4")
model = YOLO("C:/Users/tunahan/Desktop/modelTest/bestv8.pt", verbose=False).to("cuda")

tracker = None
lost_counter = 0
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # FPS hesaplama
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    h, w, _ = frame.shape
    tespit = None

    # AV Çerçeve koordinatları
    av_x_min, av_x_max = int(w * 0.25), int(w * 0.75)
    av_y_min, av_y_max = int(h * 0.10), int(h * 0.90)
    av_cx, av_cy = (av_x_min + av_x_max) // 2, (av_y_min + av_y_max) // 2
    cv2.rectangle(frame, (av_x_min, av_y_min), (av_x_max, av_y_max), (0, 255, 255), 2)

    if tracker is None:
        # İlk tespit
        results = model(frame, verbose=False)[0]
        if len(results.boxes.xyxy) > 0:
            x1, y1, x2, y2 = results.boxes.xyxy[0].cpu().numpy()
            tespit = [(x1 + x2) / 2, (y1 + y2) / 2]
            tracker = SimpleKalman(x=tespit[0], y=tespit[1])
            lost_counter = 0
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            cv2.line(frame, (int(tespit[0]), int(tespit[1])), (av_cx, av_cy), (0, 0, 0), 2)
    else:
        # Kalman tahmini
        x_kalman, y_kalman = tracker.step()

        # Tahmini çerçeve sınırları içine clamp
        x_kalman = np.clip(x_kalman, 0, w-1)
        y_kalman = np.clip(y_kalman, 0, h-1)

        # Model için crop alanı
        half = 80
        x1c = int(max(x_kalman - half, 0))
        y1c = int(max(y_kalman - half, 0))
        x2c = int(min(x_kalman + half, w-1))
        y2c = int(min(y_kalman + half, h-1))

        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            lost_counter += 1
            continue

        # Crop iyileştirme ve tespit
        crop = enhance_drone(crop, boost=2.5)
        results = model(crop, verbose=False)[0]

        if len(results.boxes.xyxy) > 0:
            bx1, by1, bx2, by2 = results.boxes.xyxy[0].cpu().numpy()
            cx = x1c + (bx1 + bx2) / 2
            cy = y1c + (by1 + by2) / 2
            tespit = [cx, cy]
            lost_counter = 0

            # Nesne kutusu (clamp ile güvenli)
            box_half = 20
            x1 = int(max(cx - box_half, 0))
            y1 = int(max(cy - box_half, 0))
            x2 = int(min(cx + box_half, w-1))
            y2 = int(min(cy + box_half, h-1))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

            # Merkezi çizgi
            cv2.line(frame, (int(cx), int(cy)), (av_cx, av_cy), (0, 0, 0), 2)
        else:
            lost_counter += 1

        # Güncellenmiş Kalman çizimi
        x_kalman, y_kalman = tracker.step(tespit)
        x_kalman = np.clip(x_kalman, 0, w-1)
        y_kalman = np.clip(y_kalman, 0, h-1)
        cv2.circle(frame, (int(x_kalman), int(y_kalman)), 5, (255, 0, 0), -1)

        # Kayıp sayacı kontrolü
        if lost_counter > 7:
            tracker = None
            lost_counter = 0

    # Gösterim
    cv2.imshow("YOLOv8 + Kalman Tracking + AV Çerçeve", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
