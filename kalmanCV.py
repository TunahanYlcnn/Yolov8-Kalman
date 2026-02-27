import cv2
import numpy as np
from ultralytics import YOLO

# Sade Kalman filtre sınıfı
class SimpleKalman:
    def __init__(self, x=0, y=0, dt=1/30, std_x=0.001, std_y=0.001):
        self.dt = dt
        # Durum: [x, x_vel, x_acc, y, y_vel, y_acc]
        self.S = np.array([x, 0, 0, y, 0, 0], dtype=float)
        # Geçiş matrisi
        self.F = np.array([
            [1, dt, 0.5*dt**2, 0, 0, 0],
            [0, 1, dt,         0, 0, 0],
            [0, 0, 1,          0, 0, 0],
            [0, 0, 0,          1, dt, 0.5*dt**2],
            [0, 0, 0,          0, 1, dt],
            [0, 0, 0,          0, 0, 1]
        ], dtype=float)
        # Ölçüm matrisi: yalnızca x ve y gözlemleniyor
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=float)
        # Ölçüm gürültü kovaryansı
        self.R = np.diag([std_x**2, std_y**2])
        # Süreç gürültü kovaryansı
        self.Q = np.diag([0.01, 0.05, 0.1, 0.01, 0.05, 0.1])
        # Başlangıç belirsizliği
        self.P = np.diag([2, 2, 10, 10, 10, 10])
        self.I = np.eye(6)

    def step(self, z=None):
        # Predict
        S_pred = self.F @ self.S
        P_pred = self.F @ self.P @ self.F.T + self.Q

        if z is not None:
            # Update
            y = np.array(z, dtype=float).reshape(2,) - (self.H @ S_pred)
            SHT = P_pred @ self.H.T
            K = SHT @ np.linalg.inv(self.H @ SHT + self.R)
            self.S = S_pred + K @ y
            self.P = (self.I - K @ self.H) @ P_pred
        else:
            self.S = S_pred
            self.P = P_pred

        return float(self.S[0]), float(self.S[3])  # x, y

# Video ve model ayarları
cap = cv2.VideoCapture("C:/Users/tunahan/Desktop/modelTest/dog2.mp4")
model = YOLO("C:/Users/tunahan/Desktop/modelTest/bestv8.pt", verbose=False)
tracker = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Ortalama bölge çizimi (yeşil)
    av_x_min, av_x_max = int(w * 0.25), int(w * 0.75)
    av_y_min, av_y_max = int(h * 0.10), int(h * 0.90)
    av_cx, av_cy = (av_x_min + av_x_max) // 2, (av_y_min + av_y_max) // 2
    cv2.rectangle(frame, (av_x_min, av_y_min), (av_x_max, av_y_max), (0, 255, 255), 2)

    # YOLOv8 tespiti
    results = model(frame, verbose=False)[0]
    detection = None
    if results.boxes:
        x1, y1, x2, y2 = results.boxes.xyxy[0].cpu().numpy().astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        detection = [cx, cy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(frame, (cx, cy), (av_cx, av_cy), (0, 0, 0), 2)

    # Kalman filtresi ile takip
    if tracker is None and detection is not None:
        tracker = SimpleKalman(x=detection[0], y=detection[1])
    elif tracker is not None:
        x_kalman, y_kalman = tracker.step(detection)
        cv2.circle(frame, (int(x_kalman), int(y_kalman)), 5, (255, 0, 0), -1)


    cv2.imshow("YOLOv8 + Kalman Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()