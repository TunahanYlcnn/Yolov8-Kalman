import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ===============================
# Kalman Takipçi Sınıfı
# ===============================
class Track:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R *= 10
        self.kf.P *= 1000
        self.kf.Q *= 0.01

        self.kf.x[:4] = np.array(bbox).reshape(4, 1)
        self.id = Track.count
        Track.count += 1
        self.hits = 0
        self.no_losses = 0

    def predict(self):
        self.kf.predict()
        return self.kf.x[:4].reshape(-1)

    def update(self, bbox):
        self.kf.update(np.array(bbox).reshape(-1, 1))
        self.hits += 1
        self.no_losses = 0


# ===============================
# Yardımcı Fonksiyonlar
# ===============================
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    ovr = inter / (area1 + area2 - inter + 1e-5)
    return ovr

def associate(detections, trackers, iou_threshold=0.25):
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_idx = linear_sum_assignment(-iou_matrix)
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_trackers = list(range(len(trackers)))

    for m in zip(*matched_idx):
        if iou_matrix[m[0], m[1]] > iou_threshold:
            matches.append(m)
            unmatched_detections.remove(m[0])
            unmatched_trackers.remove(m[1])
    return matches, unmatched_detections, unmatched_trackers


# ===============================
# Ana Takip Döngüsü
# ===============================
def run(video_source=0, model_path='bestv8.pt'):
    cap = cv2.VideoCapture(video_source)
    model = YOLO(model_path)
    tracks = []

    width = 640
    height = 360

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        results = model(frame, verbose=False)[0]
        detections = []

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for det, conf in zip(boxes, confs):
            x1, y1, x2, y2 = det[:4]
            if conf > 0.3:
                detections.append([x1, y1, x2, y2])

        predicted_boxes = [trk.predict() for trk in tracks]
        matches, unmatched_dets, unmatched_trks = associate(detections, predicted_boxes)

        for match in matches:
            det = detections[match[0]]
            tracks[match[1]].update(det)

        for idx in unmatched_dets:
            tracks.append(Track(detections[idx]))

        new_tracks = []
        for trk in tracks:
            trk.no_losses += 1
            if trk.no_losses < 5:
                new_tracks.append(trk)
        tracks = new_tracks

        for trk in tracks:
            x1, y1, x2, y2 = trk.kf.x[:4].reshape(-1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {trk.id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 + Kalman + ByteTrack", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# Başlat
# ===============================
if __name__ == "__main__":
    run(video_source="C:/Users/tunahan/Desktop/sıfırdanRe3.1/dog2.mp4",
        model_path="C:/Users/tunahan/Desktop/sıfırdanRe3.1/bestv8.pt")
