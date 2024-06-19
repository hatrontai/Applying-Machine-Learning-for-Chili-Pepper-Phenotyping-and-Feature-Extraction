import numpy as np
import math
import cv2
from pyzbar.pyzbar import decode
from pathlib import Path


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

    def get_wrinkle(self, ep= 0.005):
        mask_fruit = self.mask
        contour,hierarchy  = cv2.findContours(mask_fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_line = np.array(contour[0])

        length_L1 = cv2.arcLength(contour_line, True)
        appox_contour = cv2.approxPolyDP(contour_line, ep*length_L1, True)
        length_L2 = cv2.arcLength(appox_contour, True)
        smoothness = length_L2/length_L1
        wrinkle = round(smoothness, 5)


        # wrinkle = 0
        # for i in range(len(contour_line)):
        #     angle, d1, d2 = calculate_angle(contour_line[i-2][0], contour_line[i-1][0], contour_line[i][0])
        #     wrinkle += angle
        # wrinkle = round(wrinkle/len(contour_line), 2)

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
    
    def get_contour(self, ep= 0.005):
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_line = np.zeros_like(self.mask)
        cv2.drawContours(contour_line, contours, -1, (255, 255, 255), 3)
        appox_contour = cv2.approxPolyDP(contours[0], ep*cv2.arcLength(contours[0], True), True)
        cv2.drawContours(contour_line, [appox_contour], -1, (255, 255, 255), 3, cv2.LINE_AA)

        return contour_line