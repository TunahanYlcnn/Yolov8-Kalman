import cv2
import numpy as np
from ultralytics import YOLO
import time

def soft_nms(boxes, scores, iou_thr=0.5, sigma=0.5, score_thr=0.001, method='gaussian'):
    boxes = boxes.copy()
    scores = scores.copy()
    N = len(scores)
    for i in range(N):
        max_idx = i + np.argmax(scores[i:])
        # swap
        boxes[i], boxes[max_idx] = boxes[max_idx].copy(), boxes[i].copy()
        scores[i], scores[max_idx] = scores[max_idx], scores[i]
        for j in range(i+1, N):
            # compute IoU
            x1 = max(boxes[i,0], boxes[j,0])
            y1 = max(boxes[i,1], boxes[j,1])
            x2 = min(boxes[i,2], boxes[j,2])
            y2 = min(boxes[i,3], boxes[j,3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
            area_j = (boxes[j,2]-boxes[j,0])*(boxes[j,3]-boxes[j,1])
            iou = inter / (area_i + area_j - inter + 1e-6)
            # update score
            if method=='linear' and iou > iou_thr:
                scores[j] *= (1 - iou)
            elif method=='gaussian':
                scores[j] *= np.exp(-(iou*iou)/sigma)
    keep = scores > score_thr
    return boxes[keep], scores[keep]

def enhance_drone(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l) ###
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

class SimpleKalman:
    def __init__(self, x=0, y=0, dt=1/30):
        self.dt = dt
        self.S = np.array([x, 0, 0, y, 0, 0], dtype=float)
        F = np.eye(6)
        F[0,1] = F[3,4] = dt
        F[0,2] = F[3,5] = 0.5*dt*dt
        F[1,2] = F[4,5] = dt
        self.F = F
        self.H = np.zeros((2,6)); self.H[0,0]=1; self.H[1,3]=1
        self.R = np.eye(2) * 0.01
        self.Q = np.diag([0.01,0.02,0.05,0.01,0.02,0.05])
        self.P = np.eye(6) * 10
        self.I = np.eye(6)
    def step(self, z=None):
        S_pred = self.F @ self.S
        P_pred = self.F @ self.P @ self.F.T + self.Q
        if z is not None:
            y = np.array(z) - self.H @ S_pred
            K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)
            self.S = S_pred + K @ y
            self.P = (self.I - K @ self.H) @ P_pred
        else:
            self.S, self.P = S_pred, P_pred
        return self.S[0], self.S[3]

cap = cv2.VideoCapture("C:/Users/tunahan/Desktop/modelTest/dog2.mp4")
model = YOLO("C:/Users/tunahan/Desktop/modelTest/bestv8.pt", verbose=False).to("cuda")
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

tracker = None
lost_counter = 0
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        x,y,ww,hh = cv2.boundingRect(c)
        roi = (x, y, x+ww, y+hh)
    prev_gray = gray

    tta_scales = [1.0, 0.75, 1.25]

    tespit = None
    if tracker is None or roi is None:
        results = model(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        boxes, scores = soft_nms(boxes, scores)
        if len(scores) > 0:
            x1, y1, x2, y2 = boxes[np.argmax(scores)]
            cx, cy = (x1+x2)/2, (y1+y2)/2
            tracker = SimpleKalman(cx, cy)
            lost_counter = 0
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255),2)
    else:
        xk, yk = tracker.step()
        xk, yk = np.clip([xk, yk], [0,0], [w-1, h-1])
        half = 80
        x1c, y1c = int(max(xk-half,0)), int(max(yk-half,0))
        x2c, y2c = int(min(xk+half,w-1)), int(min(yk+half,h-1))
        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size>0:
            crop = enhance_drone(crop)
            # Multi-scale TTA
            all_boxes, all_scores = [], []
            for s in tta_scales:
                resized = cv2.resize(crop, None, fx=s, fy=s)
                res = model(resized)[0]
                b = res.boxes.xyxy.cpu().numpy()/s
                scr = res.boxes.conf.cpu().numpy()
                all_boxes.append(b)
                all_scores.append(scr)
            boxes = np.vstack(all_boxes)
            scores = np.concatenate(all_scores)
            boxes, scores = soft_nms(boxes, scores)
            if len(scores)>0:
                bx1,by1,bx2,by2 = boxes[np.argmax(scores)]
                cx = x1c + (bx1+bx2)/2
                cy = y1c + (by1+by2)/2
                tespit = [cx, cy]
                lost_counter = 0
                x1,y1 = int(cx-20), int(cy-20)
                x2,y2 = int(cx+20), int(cy+20)
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
        else:
            lost_counter +=1

        xk, yk = tracker.step(tespit)
        cv2.circle(frame, (int(xk),int(yk)),5,(255,0,0),-1)
        if lost_counter > 7:
            tracker=None; lost_counter=0

    curr_time = time.time()
    fps = 1/(curr_time-prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
    cv2.imshow("Robust Drone Detection", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
