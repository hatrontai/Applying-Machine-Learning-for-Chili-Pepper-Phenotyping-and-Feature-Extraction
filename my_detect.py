import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import math
from pyzbar.pyzbar import decode
import torch.nn as nn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def segm(img, threshold=56):
    # print(type(img))
    img = img.copy()
    img = np.array(img)
    img = np.where(img >= threshold, 255, 0)

    return img

def calculate_angle(p1, p2, p3):
    
    # Tính các véc-tơ
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Tính độ lớn của các véc-tơ
    norm_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    norm_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Tính tích vô hướng của hai véc-tơ
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Tính góc bằng công thức tích vô hướng
    angle = math.acos(dot_product / (norm_v1 * norm_v2))
    # angle_deg = angle_rad * 180 / math.pi

    return angle, norm_v1, norm_v2

def find_centroid(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    
    non_zero_points = []
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] > 0:
                non_zero_points.append((j, i))  # Lưu ý thứ tự (x, y)
    
    if not non_zero_points:

        return None
    
    centroid_x = sum(x for x, y in non_zero_points) / len(non_zero_points)
    centroid_y = sum(y for x, y in non_zero_points) / len(non_zero_points)
    
    return centroid_x, centroid_y


def korean_imread(filename, flags= cv2.IMREAD_COLOR, dtype= np.uint8):
    try:
        n = np.fromfile(filename, dtype= dtype)
        img = cv2.imdecode(n, flags)

        return img
    
    except Exception as e:
        print(e)

        return None
    

def qr_code(source):
    org = korean_imread(source)
    gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    qr_codes = decode(gray)
    if qr_codes:
        qr_code = qr_codes[0]

        x,y,w,h = qr_code.rect

        try:
            SF = 10/((w+h)/2)
            # print("ratio (mm/pixel): ", SF)

            return SF, y+h
        
        except Exception as e:
            print("Error:", e)

            return None


def cal_number_seeds(bbox, center_seeds):
    center_seeds = np.array(center_seeds)
    if len(center_seeds) == 0:
        return 0
    mask = (center_seeds[:, 0] >= int(bbox[0])) & (center_seeds[:, 0] <= int(bbox[2])) & (center_seeds[:, 1] >= int(bbox[1])) & (center_seeds[:, 1] <= int(bbox[3]))
    points_in_bbox = center_seeds[mask]
    number_seeds = len(points_in_bbox)
    return number_seeds


class Fruit(object):

    def __init__(self, source, id, bbox):
        self.id = id 
        self.source = str(source)
        self.bbox = bbox
        self.ratio,_ = qr_code(source)
        self.name = self.get_name()
        self.img = self.get_img()
        self.width_bbox, self.height_bbox = self.get_width_height_bbox()
        self.mask = self.get_mask()
        self.number_seeds = 0
        self.width_fruit, self.length_fruit = self.get_width_length_fruit()
        self.area_fruit = self.get_area()
        self.redness = self.get_redness()
        self.wrinkle = self.get_wrinkle()

    def get_img(self):
        img = cv2.imread(self.source)
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox = self.bbox
        fruit = img_RGB[int(bbox[1]-50):int(bbox[3]+50), int(bbox[0]-50):int(bbox[2]+50)]
        # cv2.imshow("1", fruit)
        # cv2.waitKey(0)
        return fruit
    
    def get_mask(self, use_closing = True, use_contour = True):
        img = self.img
        # cv2.imshow("", img)
        # cv2.imshow("2", img)
        
        img_red = np.array(img[:, :, 0])
        hist = cv2.calcHist([img_red], [0], None, [256], [0, 256])

        fruit_mask = segm(img_red, hist.argmax() + 16)
        fruit_mask = fruit_mask.astype(np.uint8)
        # cv2.imshow("3", fruit_mask)

        if use_closing:
            kernel = np.ones((10, 10), np.uint8)
            fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            fruit_mask = fruit_mask.astype(np.uint8)
        # cv2.imshow("4", fruit_mask)
        if use_contour:
            contours, hierarchy = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_areas = [cv2.contourArea(cnt) for cnt in contours]
            max_area = max(contour_areas)
            fruit_mask = np.zeros_like(fruit_mask)
            contour_line = np.zeros_like(fruit_mask)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= max_area:
                    cv2.drawContours(fruit_mask, [cnt], -1, (255, 255, 255), -1)
 
        return fruit_mask

    def get_name(self):
        source = Path(self.source)
        name = source.stem
        return f"{name}_{self.id}.jpg"

    def get_width_height_bbox(self):
        img = self.img
        bbox = self.bbox
        ratio = self.ratio
        width_bbox = round((bbox[2] - bbox[0])*ratio, 2)
        height_bbox = round((bbox[3] - bbox[1])*ratio, 2)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        low_green = np.array([25, 20, 20])
        high_green  = np.array([90, 255, 255])

        green_mask = cv2.inRange(hsv_img, low_green, high_green)
        centroid_pedicel = find_centroid(green_mask)
        if centroid_pedicel != None:
            d1 = np.abs(len(green_mask) - 2*int(centroid_pedicel[1]))
            d2 = np.abs(len(green_mask[0]) - 2*int(centroid_pedicel[0]))
            # print(green_mask.shape, centroid_pedicel, d1, d2)
            if d1 < d2:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                self.img = img
                temp = width_bbox
                width_bbox = height_bbox
                height_bbox = temp

        return width_bbox, height_bbox
    
    def get_width_length_fruit(self):
        mask = self.mask
        top = 0
        bottom = 0
        for i in range(len(mask)):
            temp = mask[i]
            if temp.sum() > 0:
                if top == 0:
                    top = i
                bottom = i

        d_pin = int((bottom - top)/9)
        pins = []
        for i in range(9):
            pins.append(top+i*d_pin)
        pins.append(bottom)
        points = []
        length_fruit = 0
        width_fruit = 0
        for i in pins:
            temp = mask[i]
            max_j = -1
            min_j = -1
            for j in range(len(temp)):
                if temp[j]>0:
                    if min_j ==-1:
                        min_j = j
                    max_j = j
            point = [i, min_j, max_j]
            if i > top and i < bottom:
                width_fruit += max_j - min_j
            points.append(point)

        width_fruit /= 8
        for i in range(len(points)):
            if i == 0:
                continue
            p1 = [points[i-1][0], (points[i-1][1] + points[i-1][2])/2]
            p2 = [points[i][0], (points[i][1] + points[i][2])/2]
            length_fruit += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

        ratio = self.ratio
        width_fruit = round(width_fruit*ratio, 2)
        length_fruit = round(length_fruit*ratio, 2)
            
        return width_fruit, length_fruit
    
    def get_area(self):
        mask = self.mask/255
        ratio = self.ratio
        area = round(mask.sum()*ratio*ratio, 2)

        return area


    def get_redness(self):
        mask_fruit = self.mask
        # cv2.imshow("", self.img)
        # cv2.waitKey(0)
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        hue = np.array(hsv_img[:, :, 0].astype(int))
        # print(hue.shape, mask_fruit.shape)
        hue = np.where(mask_fruit > 0, np.abs(90 - hue), 0)
        hue = hue - 60
        hue = np.where(hue >= 0, hue, 0)
        redness = round((hue.sum()/(mask_fruit > 0).sum())/30, 4)

        return redness

    def get_wrinkle(self):
        mask_fruit = self.mask
        contour_line,_ = cv2.findContours(mask_fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_line = np.array(contour_line[0])

        wrinkle = 0
        for i in range(len(contour_line)):
            angle, d1, d2 = calculate_angle(contour_line[i-2][0], contour_line[i-1][0], contour_line[i][0])
            wrinkle += angle
        wrinkle = round(wrinkle/len(contour_line), 2)

        return wrinkle

    def get_hist(self):
        img = self.img
        img_red = np.array(img[:, :, 0])
        hist = cv2.calcHist([img_red], [0], None, [256], [0, 256])
        height = 400
        width = 800
        hist_image = np.zeros((height, width, 3), dtype=np.uint8)
        hist = cv2.normalize(hist, hist, alpha=0, beta=height, norm_type=cv2.NORM_MINMAX)
        # Draw the hist on the empty image
        for i in range(256):
            cv2.line(hist_image, (i, height), (i, int(height - hist[i])), (255, 255, 255), 1)
    
        return hist_image
    
    def get_contour(self):
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_line = np.zeros_like(self.mask)
        cv2.drawContours(contour_line, contours, -1, (255, 255, 255), 3)

        return contour_line

def detect(save_img=False):
    source, weights, view_img, save_img_para, save_txt, save_csv, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_img_para, opt.save_txt, opt.save_csv, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    chilis = []
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    print(save_dir)
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        id = 1
        print(path)
        # threshold for cut qr_code
        _, cut = qr_code(path) 
        center_seeds = []
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            (save_dir / p.stem).mkdir(parents=True, exist_ok=True)  # make dir
            save_path = str(save_dir / p.stem / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    bbox = []
                    for bbox_cuda_tensor in xyxy:
                        bbox_cpu_tensor = bbox_cuda_tensor.cpu()
                        item = bbox_cpu_tensor.numpy()
                        bbox.append(item)

                    if int(cls.item()) == 0:
                        center_seed = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
                        center_seeds.append(center_seed)

                for *xyxy, conf, cls in reversed(det):
                    bbox = []
                    for bbox_cuda_tensor in xyxy:
                        bbox_cpu_tensor = bbox_cuda_tensor.cpu()
                        item = bbox_cpu_tensor.numpy()
                        bbox.append(item)
                    
                    # except bbox detect on labels of image
                    if bbox[1] < cut:
                        continue

                    if int(cls.item()) == 1:
                        chili = Fruit(source= p, id= id, bbox= bbox)
                        chili.number_seeds = cal_number_seeds(chili.bbox, center_seeds)
                        chilis.append(chili)
                        
                        id = id + 1

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                     

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
    
    
    if save_img_para:
        for chili in chilis:
            source = Path(chili.source)
            save_path = str(save_dir / source.stem)

            img_BGR = cv2.cvtColor(chili.img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_path}/{chili.name}", img_BGR)

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            axes[0].imshow(chili.img)
            axes[0].set_title('original image')
            axes[0].set_position([0.05, 0.1, 0.22, 0.8])
            axes[1].imshow(chili.mask)
            axes[1].set_title('chili mask')
            axes[1].set_position([0.3, 0.1, 0.22, 0.8])
            axes[2].imshow(chili.get_hist())
            axes[2].set_title('Histogram')
            axes[2].set_position([0.55, 0.1, 0.22, 0.8])
            fig.text(0.8, 0.9, f'Name image: {chili.name}', fontsize=12, color='black', bbox=None)
            fig.text(0.8, 0.8, f'Width of bbox: {chili.width_bbox} mm', fontsize=12, color='black', bbox=None)
            fig.text(0.8, 0.7, f'Height of bbox: {chili.height_bbox} mm', fontsize=12, color='black', bbox=None)
            fig.text(0.8, 0.6, f'Average width of fruit: {chili.width_fruit} mm', fontsize=12, color='black', bbox=None)
            fig.text(0.8, 0.5, f'Length of fruit: {chili.length_fruit} mm', fontsize=12, color='black', bbox=None)
            fig.text(0.8, 0.4, f'Area of fruit: {chili.area_fruit} mm^2', fontsize=12, color='black', bbox=None)
            fig.text(0.8, 0.3, f'Degree of redness (0,1): {chili.redness}', fontsize=12, color='black', bbox=None)
            fig.text(0.8, 0.2, f'Number of seeds: {chili.number_seeds}', fontsize=12, color='black', bbox=None)
            fig.text(0.8, 0.1, f'Wrinkle of fruit: {chili.wrinkle}', fontsize=12, color='black', bbox=None)

            plt.savefig(f"{save_path}/results_{chili.id}.jpg")
    if save_csv:
        columns = ['Image_name', 
                   'Number_seeds', 
                   'width_bbox',
                   'height_bbox',
                   'width_fruit',
                   'length_fruit',
                   'area_fruit',
                   'redness',
                   'wrinkle']
        results = pd.DataFrame(columns=columns)
        for chili in chilis:
            new_row = {'Image_name': chili.name, 
                       'Number_seeds': chili.number_seeds, 
                       'width_bbox': chili.width_bbox, 
                       'height_bbox': chili.height_bbox, 
                       'width_fruit': chili.width_fruit, 
                       'length_fruit': chili.length_fruit,
                       'area_fruit': chili.area_fruit, 
                       'redness': chili.redness,
                       'wrinkle': chili.wrinkle}
            results = results.append(new_row, ignore_index=True)
        csv_name = "results.csv" 
        save_path = f"{save_dir}/labels/{csv_name}"
        results.to_csv(save_path, index= False)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results to *.csv')
    parser.add_argument('--save-img-para', action='store_true', help='display results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop')
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
