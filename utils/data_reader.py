import os
import cv2
import glob

import numpy as np
import xml.etree.ElementTree as ET

from utils.labelmap import defect_values
from utils.density_map import gaussian_kernel


def indent(elem, level=0, more_sibs=False):
    i = "\n"
    if level:
        i += (level-1) * '\t'
    num_kids = len(elem)
    if num_kids:
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
            if level:
                elem.text += '\t'
        count = 0
        for kid in elem:
            indent(kid, level+1, count < num_kids - 1)
            count += 1
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
            if more_sibs:
                elem.tail += '\t'
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            if more_sibs:
                elem.tail += '\t'


def create_xml(xml_path, im, selections):
    img_addr = xml_path[:-3] + 'jpg'
    xml_addr = xml_path[:-3] + 'xml'

    im_w, im_h, im_d = im.shape

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder')
    ET.SubElement(annotation, 'filename')
    ET.SubElement(annotation, 'path')

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(im_w)
    ET.SubElement(size, 'height').text = str(im_h)
    ET.SubElement(size, 'depth').text = str(im_d)

    ET.SubElement(annotation, 'segmented').text = '0'
    tree = ET.ElementTree(annotation)
    tree.write(xml_addr)

    tree = ET.parse(xml_addr)
    root = tree.getroot()

    filename = os.path.basename(img_addr)
    abs_path = os.path.abspath(img_addr)
    splitted = os.path.split(abs_path)[0].split('\\')
    last = splitted[len(splitted) - 1]

    root.find('folder').text = last
    root.find('filename').text = filename
    root.find('path').text = abs_path

    for selection in selections:
        obj = ET.SubElement(root, 'object')

        ET.SubElement(obj, 'name').text = 'cafe'
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')

        xmin, ymin, xmax, ymax = selection

        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    indent(root)
    tree.write(xml_addr)


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


def generate_dmap(image, bboxes):
    im_h, im_w = image.shape
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


def prepare_image(image, final_size):
    scale = final_size / min(image.shape[0], image.shape[1])
    return cv2.resize(src=image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def cut_image(image, channels=1):
    cut = min(image.shape[0], image.shape[1])

    dx = abs(cut - image.shape[1]) // 2
    dy = abs(cut - image.shape[0]) // 2

    image = image[dy:dy+cut, dx:dx+cut]
    return np.reshape(image, (cut, cut, channels))


def load(dirs, final_size=256):
    data = []
    for _dir in dirs:
        addrs = glob.glob(os.path.join(_dir, '*.xml'))
        for addr in addrs:
            print(f'Loading data from: {addr}')

            image, bboxes = read_xml(addr)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = prepare_image(image, final_size)
            dmap = generate_dmap(image, bboxes)

            image = cut_image(image)
            dmap = cut_image(dmap)

            data.append([image, dmap])

    return data


def load_images(dirs, final_size=256):
    data = []
    for _dir in dirs:
        addrs = glob.glob(os.path.join(_dir, '*.jpg'))
        for addr in addrs:
            print(f'Loading image: {addr}')

            image = cv2.imread(addr)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = prepare_image(image, final_size)
            image = cut_image(image)
            data.append(image)

    return data
