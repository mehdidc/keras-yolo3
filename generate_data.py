import pandas as pd
from skimage.io import imread
from clize import run


def read_label_map(filename):
    fd = open(filename)
    lines = fd.readlines()
    d = {}
    for i in range(0, len(lines), 5):
        _, name, id_, display_name, _ = lines[i:i+5]
        name = name.replace('name:', '').replace('"', '').strip()
        display_name = display_name.replace('display_name:', '').replace('"', '').strip()
        id_ = int(id_.replace('id:', '').strip())
        d[display_name] = name
    return d

def generate(split, *, label='Beer', out='generated_data'):
    df = pd.read_csv('data/annotations/{}.csv'.format(split))
    lm = read_label_map('data/annotations/label_map.pbtxt')
    df = df[df['LabelName'] == lm[label]]
    lines = []
    for image_id, bbs in df.groupby('ImageID'):
        filename = 'data/images/{}/{}.jpg'.format(split, image_id)
        im = imread(filename)
        if len(im.shape) != 3:
            continue
        h, w = im.shape[0:2]
        xmins = bbs['XMin'].values * w
        xmaxs = bbs['XMax'].values * w
        ymins = bbs['YMin'].values * h
        ymaxs = bbs['YMax'].values * h
       
        xmins = xmins.astype('int32')
        xmaxs = xmaxs.astype('int32')
        ymins = ymins.astype('int32')
        ymaxs = ymaxs.astype('int32')
        boxes = []
        clid = 0
        for i in range(len(xmins)):
            boxes.append('{},{},{},{},{}'.format(xmins[i], ymins[i], xmaxs[i], ymaxs[i], clid))
        boxes = ' '.join(boxes)
        line = '{} {}'.format(filename, boxes)
        lines.append(line)
    lines = '\n'.join(lines)
    out = '{}/{}.txt'.format(out, split)
    with open(out, 'w') as fd:
        fd.write(lines)

if __name__ == '__main__':
    run(generate)
