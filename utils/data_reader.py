import os
import cv2
import glob
import json

import numpy as np
import xml.etree.ElementTree as ET

from utils.labelmap import defect_values
from utils.density_map import gaussian_kernel


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text

    dirname = os.path.dirname(xml_path)

    img_path = os.path.join(dirname, filename)
    image = cv2.imread(img_path)

    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text) / image.shape[1]
        xmax = float(bndbox.find('xmax').text) / image.shape[1]
        ymin = float(bndbox.find('ymin').text) / image.shape[0]
        ymax = float(bndbox.find('ymax').text) / image.shape[0]

        weight = defect_values[name]

        bbox = {
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'name': name,
            'weight': weight
        }

        bboxes.append(bbox)

    return image, bboxes


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


def generate_dmap(image, bboxes):
    im_h = image.shape[0]
    im_w = image.shape[1]

    dmap = np.zeros((im_h, im_w), np.float32)

    for bbox in bboxes:
        xmin = bbox['xmin'] * im_w
        xmax = bbox['xmax'] * im_w
        ymin = bbox['ymin'] * im_h
        ymax = bbox['ymax'] * im_h

        x = int(xmin + (xmax - xmin) / 2)
        y = int(ymin + (ymax - ymin) / 2)

        w = 100  # bbox['weight'] * 100

        s = ((xmax - xmin) + (ymax - ymin)) / 8
        dmap += gaussian_kernel(center=(x, y), map_size=(im_h, im_w), A=w, sx=s, sy=s)

    return dmap


def generate_seg(image, shapes):
    im_h = image.shape[0]
    im_w = image.shape[1]

    seg_map = np.zeros_like(image, np.float32)
    wei_map = np.ones_like(image, np.float32)

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

        cv2.fillPoly(wei_map, [points], (2, 2, 2))
        cv2.polylines(wei_map, [points], True, (10, 10, 10), thickness=3)

    return seg_map, wei_map


def prepare_image(image, final_size):
    scale = final_size / min(image.shape[0], image.shape[1])
    return cv2.resize(src=image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def cut_image(image, channels=1):
    cut = min(image.shape[0], image.shape[1])

    dx = abs(cut - image.shape[1]) // 2
    dy = abs(cut - image.shape[0]) // 2

    image = image[dy:dy+cut, dx:dx+cut]
    return np.reshape(image, (cut, cut, channels))


def load(dirs, final_size=128):
    data = []
    for _dir in dirs:
        addrs = glob.glob(os.path.join(_dir, '*.json'))
        for addr in addrs:
            print(f'Loading data from: {addr}')

            # x, bboxes = read_xml(addr)
            x, shapes = read_json(addr)

            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = prepare_image(x, final_size)

            # y = generate_dmap(image, bboxes)
            y, w = generate_seg(x, shapes)

            x = cut_image(x)
            y = cut_image(y)
            w = cut_image(w)

            data.append([x, y])

    return data


def load_images(dirs, final_size=128):
    data = []
    for _dir in dirs:
        addrs = glob.glob(os.path.join(_dir, '*.jpg'))
        for addr in addrs:
            print(f'Loading image: {addr}')

            x = cv2.imread(addr)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

            x = prepare_image(x, final_size)
            x = cut_image(x)
            data.append(x)

    return data
