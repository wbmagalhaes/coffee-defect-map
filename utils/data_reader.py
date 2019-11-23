import os
import cv2
import glob
import json

import numpy as np

from utils.labelmap import defect_values


def read_json(addr):
    with open(addr) as json_file:
        data = json.load(json_file)

        dirname = os.path.dirname(addr)
        filename = data['imagePath']

        img_path = os.path.join(dirname, filename)
        image = cv2.imread(img_path)

        shapes = data['shapes']

        def rescale_point(point, img):
            x, y = point
            x = x / image.shape[1]
            y = y / image.shape[0]
            return x, y

        def rescale_shape(shape, img):
            shape['points'] = [rescale_point(point, img) for point in shape['points']]
            return shape

        shapes = [rescale_shape(shape, image) for shape in shapes]

    return image, shapes


def generate_seg(image, shapes):
    im_h = image.shape[0]
    im_w = image.shape[1]

    seg_map = np.zeros_like(image, np.float32)

    for shape in shapes:
        def rescale_point(point):
            x, y = point
            x = int(x * im_w)
            y = int(y * im_h)
            return x, y

        points = [rescale_point(point) for point in shape['points']]
        points = np.array(points)

        cv2.fillPoly(seg_map, [points], (1, 1, 1))
        cv2.polylines(seg_map, [points], True, (0, 0, 0), thickness=2)

    return seg_map


def resize_image(image, size):
    scale = size / min(image.shape[0], image.shape[1])
    return cv2.resize(src=image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def crop_image(image, channels=1):
    size = min(image.shape[0], image.shape[1])

    dx = abs(size - image.shape[1]) // 2
    dy = abs(size - image.shape[0]) // 2

    image = image[dy:dy + size, dx:dx + size]
    return np.reshape(image, (size, size, channels))


def load_json(dirs, final_size=128):
    data = []
    for _dir in dirs:
        addrs = glob.glob(os.path.join(_dir, '*.json'))
        for addr in addrs:
            print(f'Loading data from: {addr}')

            x, shapes = read_json(addr)

            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = resize_image(x, final_size)

            y = generate_seg(x, shapes)

            x = crop_image(x)
            y = crop_image(y)

            data.append([x, y])

    return data


def split_image(image, cols=2, rows=2):
    images = []
    columns = np.hsplit(image, cols)
    for c in columns:
        imgs = np.vsplit(c, rows)
        for img in imgs:
            images.append(img)

    return np.array(images)


def prepare_image(original_img, size=256, cols=2, rows=2):
    color_img = crop_image(original_img, channels=3)
    color_img = color_img.astype(np.float32) / 255.

    grey_img = resize_image(color_img, size=size)
    grey_img = cv2.cvtColor(grey_img, cv2.COLOR_BGR2GRAY)
    grey_img = np.reshape(grey_img, (size, size, 1))

    color_imgs = split_image(color_img, cols=cols, rows=rows)
    grey_imgs = split_image(grey_img, cols=cols, rows=rows)

    return color_imgs, grey_imgs
