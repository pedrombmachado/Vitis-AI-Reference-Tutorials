import os
import time
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
from pytorch_nndct.apis import torch_quantizer
from models.common import DetectMultiBackend
from PIL import Image
import torchvision.transforms as T

def parse_yolo_labels(label_path, img_w, img_h):
    """
    Parse YOLO label file -> torch.Tensor of shape [num_objects, 5]
    Format per line: class x y w h (all normalised 0-1)
    """
    boxes = []
    labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            labels.append(int(cls))
            boxes.append([x * img_w, y * img_h, w * img_w, h * img_h])  # scale back to pixels
    if boxes:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    return boxes, labels


class CustomImageDataset(Dataset):
    def __init__(self, img_label_dir, img_size_w=640, img_size_h=640, transform=None):
        self.img_size_w = img_size_w
        self.img_size_h = img_size_h

        if transform is None:
            self.transform = T.Compose([
                T.Resize((img_size_h, img_size_w)),
                T.ToTensor(),
            ])
        else:
            self.transform = transform

        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.images = [f for f in os.listdir(img_label_dir) if f.lower().endswith(exts)]
        self.images.sort()

        self.samples = []
        for img_file in self.images:
            img_path = os.path.join(img_label_dir, img_file)
            label_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(label_path):
                self.samples.append((img_path, label_path))
            else:
                print(f"⚠️ Skipping {img_file} (no label found).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        boxes, labels = parse_yolo_labels(label_path, self.img_size_w, self.img_size_h)

        target = {
            "image_id": torch.tensor([idx]),   # quant.py expects this
            "boxes": boxes,                    # [N,4] tensor
            "labels": labels                   # [N] tensor
        }

        return image, target

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

DIVIDER = '-'*50

def quantize(build_dir, quant_mode, weights, dataset):
    quant_model = build_dir + '/quant_model'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(weights=weights)
    model = model.to(device)
    rand_in = torch.randn(1, 3, 640, 640)
    quantizer = torch_quantizer(quant_mode, model, rand_in, output_dir=quant_model)
    quantized_model = quantizer.quant_model
    quantized_model = quantized_model.to(device)

    test_dataset = CustomImageDataset(dataset, 640, 640)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    quantized_model.eval()
    
    with torch.no_grad():
        for image, target in test_loader:
            print(f'Image {target["image_id"][0][0]}')
            output = quantized_model(image.to(device))
            pred = non_max_suppression(output)
            print(pred)


    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  

def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-b',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-w',  '--weights',  type=str,  help='Path to yolo weights file')
  ap.add_argument('-d',  '--dataset',  type=str,  help='Path to your calibration directory with subdirectories called "images" and "labels"' )
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--weights    : ',args.weights)
  print ('--dataset    : ',args.dataset)
  print(DIVIDER)

  quantize(args.build_dir, args.quant_mode, args.weights, args.dataset)
  return

if __name__ == '__main__':
    run_main()
