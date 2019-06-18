import os

import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import json

import matplotlib.pyplot as plt

from utils import config
from utils import labelmap


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


def create_xml(addr, im, selections):
    img_addr = addr[:-3] + 'jpg'
    xml_addr = addr[:-3] + 'xml'

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


def read_xml(addr):
    tree = ET.parse(addr)
    root = tree.getroot()

    filename = root.find('filename').text
    print('Lendo imagem: ' + filename)

    dirname = os.path.dirname(addr)

    image = cv.imread(os.path.join(dirname, filename))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)
    image /= 255.0
    
    ysize, xsize = image.shape

    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text) / xsize
        xmax = float(bndbox.find('xmax').text) / xsize
        ymin = float(bndbox.find('ymin').text) / ysize
        ymax = float(bndbox.find('ymax').text) / ysize

        weight = labelmap.weight_of_label(name)

        label = {
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'name': name,
            'weight': weight
        }

        labels.append(label)

    return filename, image, labels


def create_json(addr, im, selections):
    img_addr = addr[:-3] + 'jpg'
    json_addr = addr[:-3] + 'json'

    filename = os.path.basename(img_addr)
    dirname = os.path.dirname(img_addr)

    im_w, im_h, im_d = im.shape

    def blob_data(selection):
        xmin, ymin, xmax, ymax = selection

        object_data = {
            "name": "unclassified",
            "weight": 0,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax
        }

        return object_data

    objects = [blob_data(selection) for selection in selections]

    data = {
        "filename": filename,
        "dirname": dirname,
        "width": im_w,
        "height": im_h,
        "depth": im_d,
        "objects": objects
    }

    with open(json_addr, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=False)


def read_json(addr):
    with open(addr) as json_file:
        print(addr)
        data = json.load(json_file)

        dirname = os.path.dirname(addr)
        filename = data['filename']

        image = cv.imread(os.path.join(dirname, filename))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255

        labels = data['objects']

    return filename, image, labels
