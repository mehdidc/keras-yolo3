from PIL import Image
import numpy as np
from collections import defaultdict
from skimage.transform import resize
from skimage.io import imread
from yolo3.yolo import YOLO
from clize import run
from joblib import dump

eps = 1e-12
iou_threshold = 0.45

def evaluate(data, model_path, anchors_path, classes_path):
    yolo =  YOLO(
        model_path=model_path,
        anchors_path=anchors_path,
        classes_path=classes_path,
        score_threshold=0.0,
        iou=iou_threshold,
        max_boxes=10000,
    )
    lines = open(data).readlines()
    B_list = []
    BP_list = []
    for i, l in enumerate(lines):
        toks = l.strip().split(' ')
        image_filename = toks[0]
        boxes = toks[1:]

        x = imread(image_filename) 
        x = Image.fromarray(x)

        B = [list(map(int, b.split(','))) for b in boxes]
        out_boxes, out_scores, out_classes = yolo.predict_image(x)
        BP = [list(tuple(b) + (c, s)) for b, s, c in zip(out_boxes, out_scores, out_classes)]
        B_list.extend(B)
        BP_list.extend(BP)
        if i % 10 == 0:
            print('[{:05d}]/[{:05d}]'.format(i, len(lines)))
        break
    stats = get_stats(B_list, BP_list)

    for k in sorted(stats.keys()):
        v = stats[k]
        print('{}: {:.2f}'.format(k, v))


def get_stats(B, BP):
    precs, recs = PR(B, BP)
    d = {}
    for th in (0.5, 0.6, 0.8, 0.9, 0.95, 0.99):
        vals = [p for p, r in zip(precs, recs) if r >= th]
        if len(vals):
            p = max(vals)
        else:
            p = 0
        vals = [r for p, r in zip(precs, recs) if p >= th]
        if len(vals):
            r = max(vals)
        else:
            r = 0
        d['prec({:.2f})'.format(th)] = p
        d['rec({:.2f})'.format(th)] = r

    bmax = max(B, key=lambda b:(b[2]-b[0]) * (b[3]-b[1]))
    detected = 0
    for i, p in enumerate(BP):
        *bp, pred_class_id, score = p
        *bt, class_id = bmax
        if iou(bp, bt) >= iou_threshold and class_id == pred_class_id:
            detected = 1
            break
    d['detected'] = detected
    return d


def PR(B, BP, iou_threshold=0.45):
    R = np.zeros(len(B))
    P = np.zeros(len(BP))
    nb_precision = 0
    nb_recall = 0
    precisions = []
    recalls = []
    BP = sorted(BP, key=lambda p:p[-1], reverse=True)
    for i, p in enumerate(BP):
        *bp, pred_class_id, score = p
        for j, t in enumerate(B):
            *bt, class_id = t
            if iou(bp, bt) >= iou_threshold and class_id == pred_class_id:
                if R[j] == 0:
                    R[j] = 1
                    nb_recall += 1
                if P[i] == 0:
                    P[i] = 1
                    nb_precision += 1
        p = nb_precision / (i + 1)
        r = nb_recall / len(B)
        precisions.append(p)
        recalls.append(r)
    return precisions, recalls

def iou(bbox1, bbox2):
    x, y, xm, ym = bbox1
    w = xm - x
    h = ym - y
    xx, yy, xxm, yym = bbox2
    ww = xxm - xx
    hh = yym - yy
    winter = min(x + w, xx + ww) - max(x, xx)
    hinter = min(y + h, yy + hh) - max(y, yy)
    if winter < 0 or hinter < 0:
        inter = 0
    else:
        inter = winter * hinter
    union = w * h + ww * hh - inter
    return inter / (union + eps)

if __name__ == '__main__':
    run(evaluate)
