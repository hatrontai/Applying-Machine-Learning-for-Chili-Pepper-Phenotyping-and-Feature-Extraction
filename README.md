# Applying-Machine-Learning-for-Chili-Pepper-Phenotyping-and-Feature-Extraction

``` shell
# Install my requirement
pip install -r requirements.txt
```

Using model [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

## Installation

``` shell
# Clone github yolov7 to maim
git clone https://github.com/augmentedstartups/yolov7.git
cd yolov7

# Install requirement of yolov7
pip install -r requirements.txt
```
Move file object.py to utils and my_detect.py to main
- yolov7
  - my_detect.py
  - utils
    - object.py 

## Detect
``` shell
python my_detect.py --weights weight/best.pt --source inference/images/chili.jpg --save-txt --save-conf --save-csv --save-img-para
```
