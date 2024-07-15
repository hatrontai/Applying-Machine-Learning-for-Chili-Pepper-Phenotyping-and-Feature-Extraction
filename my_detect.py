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
from utils.object import qr_code, Fruit, cal_number_seeds


def detect(save_img=False):
    source, weights, view_img, save_img_para, save_txt, save_csv, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_img_para, opt.save_txt, opt.save_csv, opt.img_size, not opt.no_trace
    save_sample = True
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    chilis = []
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_csv else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
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
        print(path)
        # threshold for cut qr_code
        _, cut = qr_code(path) 
        center_seeds = []
        center_chilis = []
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
            (save_dir / p.stem if save_img or save_img_para else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
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
                    else:
                        if bbox[1] >= cut:
                            center_chili = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
                            center_chilis.append(center_chili)

                center_chilis = sorted(center_chilis, key=lambda x: sum(x))
                # print(center_chilis)

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
                        indices  = [index for index, center in enumerate(center_chilis) if center[0]== (bbox[0]+bbox[2])/2 and center[1]== (bbox[1]+bbox[3])/2]
                        id = indices[0] + 1
                        chili = Fruit(source= p, id= id, bbox= bbox)
                        chili.number_seeds = cal_number_seeds(chili.bbox, center_seeds)
                        chilis.append(chili)

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

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            axes[0].imshow(chili.img)
            axes[0].set_title('original image')
            axes[0].set_position([0.05, 0.1, 0.22, 0.8])
            axes[1].imshow(chili.mask)
            axes[1].set_title('chili mask')
            axes[1].set_position([0.3, 0.1, 0.22, 0.8])
            axes[2].imshow(chili.get_contour())
            axes[2].set_title('contours')
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

    if save_sample:
        for chili in chilis:
            source = Path(chili.source)
            save_path = str(save_dir / source.stem)

            img_BGR = cv2.cvtColor(chili.img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_path}/{chili.name}", img_BGR)
            mask_segm = chili.get_mask(use_closing = False, use_contour = False)
            cv2.imwrite(f"{save_path}/mask_segm_{chili.id}.jpg", mask_segm)
            mask_closing = chili.get_mask(use_contour = False)
            cv2.imwrite(f"{save_path}/mask_closing_{chili.id}.jpg", mask_closing)
            mask = chili.get_mask()
            cv2.imwrite(f"{save_path}/mask_{chili.id}.jpg", mask)
            hist = chili.get_hist()
            cv2.imwrite(f"{save_path}/histogram_{chili.id}.jpg", hist)


    if save_csv:
        columns = ['Image_name',
                   'kind', 
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
                       'kind': chili.name[4:12],
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
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
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
