# -------------------- Imports --------------------
import os
import time
import glob
import tarfile
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from ultralytics import YOLO

# -------------------- Constants --------------------
USE_SMALL_NET = False
CROP_SIZE = 227
CROP_PAD = 2
LOG_DIR = "C:/Users/tunahan/Desktop/sıfırdanRe3.1/logs"
GPU_ID = "0"

# -------------------- Utility: PyTorch --------------------
numpy_dtype_to_pytorch_dtype_warn = False
from_numpy_warn = defaultdict(lambda: False)

def setup_devices(devices):
    if not torch.cuda.is_available():
        raise Exception("Cuda not found")
    ids = [int(gpu_id.strip()) for gpu_id in str(devices).split(",")]
    return [f"cuda:{i}" for i in ids]

def from_numpy(np_array):
    np_array = np.asarray(np_array)
    if np_array.dtype == np.uint32:
        np_array = np_array.astype(np.int32)
    elif np_array.dtype == np.dtype('O') or np_array.dtype.type == np.str_:
        return np_array
    return torch.from_numpy(np_array)

def to_numpy_array(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    elif isinstance(array, dict):
        return {k: to_numpy_array(v) for k, v in array.items()}
    else:
        return np.asarray(array)

def restore(net, save_file, saved_variable_prefix="", new_variable_prefix="", skip_filter=None):
    try:
        with torch.no_grad():
            net_state = net.state_dict()
            restore_dict = torch.load(save_file, map_location='cpu')
            for name, param in restore_dict.items():
                key = new_variable_prefix + name[len(saved_variable_prefix):]
                if skip_filter and not skip_filter(key):
                    continue
                if key in net_state and net_state[key].size() == param.size():
                    net_state[key].copy_(param)
    except Exception as e:
        print(f"Warning: could not restore from {save_file}: {e}")

def restore_from_folder(net, folder, *args, **kwargs):
    files = sorted(glob.glob(os.path.join(folder, '*.pt')), key=os.path.getmtime)
    if files:
        restore(net, files[-1], *args, **kwargs)

# -------------------- Bounding Box Utilities --------------------
LIMIT = 99999999

def clip_bbox(bboxes, minC, maxW, maxH):
    b = bboxes.copy()
    b[[0,2]] = np.clip(b[[0,2]], minC, maxW)
    b[[1,3]] = np.clip(b[[1,3]], minC, maxH)
    return b

def xyxy_to_xywh(b):
    x1,y1,x2,y2 = b
    xc, yc = (x1+x2)/2, (y1+y2)/2
    w, h = x2-x1, y2-y1
    return np.array([xc,yc,w,h])

def xywh_to_xyxy(b):
    xc,yc,w,h = b
    x1,y1 = xc-w/2, yc-h/2
    x2,y2 = xc+w/2, yc+h/2
    return np.array([x1,y1,x2,y2])

def scale_bbox(bboxes, scalars):
    xc,yc,w,h = xyxy_to_xywh(bboxes)
    sw,sh = scalars if isinstance(scalars,(list,tuple,np.ndarray)) else (scalars,scalars)
    w2,h2 = w*sw, h*sh
    return xywh_to_xyxy([xc,yc,w2,h2])

def to_crop_coordinate_system(b, crop_loc, pad, size):
    cl = scale_bbox(crop_loc, pad)
    cw,ch = xyxy_to_xywh(cl)[2:]
    return (b - cl[[0,1,0,1]]) * size / np.array([cw,ch,cw,ch])

def from_crop_coordinate_system(b, crop_loc, pad, size):
    cl = scale_bbox(crop_loc, pad)
    cw,ch = xyxy_to_xywh(cl)[2:]
    return b * np.array([cw,ch,cw,ch]) / size + cl[[0,1,0,1]]

# -------------------- Image Utilities --------------------
def get_cropped_input(img, bbox, padScale, outputSize):
    bbox = np.array(bbox, dtype=np.float32)
    h,w = img.shape[:2]
    x1,y1,x2,y2 = bbox
    cx,cy = (x1+x2)/2, (y1+y2)/2
    bw,bh = (x2-x1)*padScale, (y2-y1)*padScale
    x0,y0 = int(round(cx-bw/2)), int(round(cy-bh/2))
    patch = img[max(y0,0):min(y0+int(bh),h), max(x0,0):min(x0+int(bw),w)]
    if patch.size == 0:
        patch = np.zeros((outputSize,outputSize,3),dtype=img.dtype)
    else:
        patch = cv2.resize(patch,(outputSize,outputSize))
    return patch, np.array([x0,y0,x0+int(bw),y0+int(bh)],dtype=np.float32)

# -------------------- Model Definitions --------------------
class CaffeLSTMCell(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.block_input = nn.Linear(input_size+output_size, output_size)
        self.input_gate  = nn.Linear(input_size+output_size*2, output_size)
        self.forget_gate = nn.Linear(input_size+output_size*2, output_size)
        self.output_gate = nn.Linear(input_size+output_size*2, output_size)

    def forward(self, inputs, hx=None):
        if hx is None:
            zeros = torch.zeros(inputs.size(0), self.block_input.out_features, device=inputs.device)
            hx = (zeros, zeros)
        h_prev, c_prev = hx
        concat = torch.cat([inputs, h_prev], dim=1)
        block = torch.tanh(self.block_input(concat))
        peephole = torch.cat([concat, c_prev], dim=1)
        i = torch.sigmoid(self.input_gate(peephole)) * block
        f = torch.sigmoid(self.forget_gate(peephole)) * c_prev
        c_new = i + f
        o = torch.sigmoid(self.output_gate(torch.cat([concat, c_new], dim=1)))
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

class Re3Net(nn.Module):
    def __init__(self, device, lstm_size=1024):
        super().__init__()
        self.device = device
        self.lstm_size = lstm_size
        self.conv = nn.Sequential(
            nn.Conv2d(6, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc6 = nn.Linear(256*6*6, 2048)
        self.lstm1 = CaffeLSTMCell(2048, lstm_size)
        self.lstm2 = CaffeLSTMCell(2048 + lstm_size, lstm_size)
        self.fc_output = nn.Linear(lstm_size, 4)
        self.lstm_state = None

    def forward(self, input, lstm_state=None):
        x = input.to(self.device).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = F.elu(self.fc6(x))

        if lstm_state is None:
            h1, c1 = self.lstm1(x)
            h2, c2 = self.lstm2(torch.cat((x, h1), 1))
        else:
            h1_p, c1_p, h2_p, c2_p = lstm_state
            h1, c1 = self.lstm1(x, (h1_p, c1_p))
            h2, c2 = self.lstm2(torch.cat((x, h1), 1), (h2_p, c2_p))

        self.lstm_state = (h1, c1, h2, c2)
        return self.fc_output(h2)

# -------------------- Tracker --------------------
class Re3Tracker(object):
    def __init__(self, gpu_id=0, model_path=None):
        dev = setup_devices(gpu_id)[0]
        self.network = Re3Net(dev)
        if model_path:
            restore(self.network, model_path)
        else:
            print("model bulunamadı")

        self.network.to(dev)
        self.network.eval()
        self.tracked_data = {}

    def track(self, uid, image, starting_box=None):
        if isinstance(image, str):
            frame = cv2.imread(image)[:, :, ::-1]
        else:
            frame = image.copy()

        if starting_box is not None:
            lstm_state = None
            prev_bbox = np.array(starting_box, dtype=np.float32)
            prev_image = frame
        elif uid in self.tracked_data:
            lstm_state, prev_bbox, prev_image = self.tracked_data[uid]
        else:
            raise Exception(f"No initial bounding box for {uid}")

        crop0, pad0 = get_cropped_input(prev_image, prev_bbox, CROP_PAD, CROP_SIZE)
        crop1, _    = get_cropped_input(frame,       prev_bbox, CROP_PAD, CROP_SIZE)
        inp = np.concatenate([crop0, crop1], axis=2)  # [227, 227, 6]
        inp = from_numpy(inp.astype(np.float32)).unsqueeze(0) / 255.0  # [1, 227, 227, 6]

        raw = self.network(inp, lstm_state)
        raw = to_numpy_array(raw.squeeze()) / 10.0
        lstm_state = self.network.lstm_state

        new_bbox = from_crop_coordinate_system(raw, pad0, CROP_PAD, CROP_SIZE)
        self.tracked_data[uid] = (lstm_state, new_bbox, frame)
        return new_bbox

# -------------------- Main --------------------
if __name__ == '__main__':
    cap = cv2.VideoCapture("C:/Users/tunahan/Desktop/tekKodRe3/dog2.mp4")
    yolo = YOLO("C:/Users/tunahan/Desktop/tekKodRe3/bestv8.pt")
    tracker = Re3Tracker(gpu_id="0", model_path="C:/Users/tunahan/Desktop/tekKodRe3/params.pt")
    initialized = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with torch.no_grad():
            res = yolo(frame, verbose=False)[0]
        if len(res.boxes.xyxy) > 0:
            x1, y1, x2, y2 = res.boxes.xyxy[0].cpu().numpy()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            if not initialized:
                bbox = tracker.track('obj', frame[:, :, ::-1], [x1, y1, x2, y2])
                initialized = True
        if initialized:
            bbox = tracker.track('obj', frame[:, :, ::-1])
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
