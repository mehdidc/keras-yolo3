from PIL import Image
import numpy as np
from collections import defaultdict
from skimage.transform import resize
from skimage.io import imread
from skimage.io import imsave
from yolo3.yolo import YOLO
from clize import run
from joblib import dump

def test(
    filename, 
    model_path, 
    anchors_path, 
    classes_path, 
    *, 
    score_threshold=0.2, 
    iou=0.45, 
    max_boxes=100
):
    yolo =  YOLO(
        model_path=model_path,
        anchors_path=anchors_path,
        classes_path=classes_path,
        score_threshold=score_threshold,
        iou=iou,
        max_boxes=max_boxes,
    )
    x = imread(filename) 
    x = Image.fromarray(x)
    x = yolo.detect_image(x)
    x = np.array(x)
    imsave('out.png', x)


if __name__ == '__main__':
    run(test)
