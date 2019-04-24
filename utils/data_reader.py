import os

import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import json

import matplotlib.pyplot as plt

from utils import config
from utils import labelmap

from utils.density_map import gaussian_kernel

def read_xml(addr):
    tree = ET.parse(addr)
    root = tree.getroot()

    filename = root.find('filename').text
    print(filename)
    
    dirname = os.path.dirname(addr)
    
    image = cv.imread(os.path.join(dirname, filename))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.uint8)

    ysize, xsize, _ = image.shape

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


def create_json(addr, blobs, xoffset, yoffset, scale):
    img_addr = addr[:-3] + 'jpg'
    json_addr = addr[:-3] + 'json'

    image = cv.imread(img_addr)
    height, width, _ = image.shape

    filename = os.path.basename(img_addr)
    dirname = os.path.dirname(img_addr)

    def blob_data(blob):
        y, x, r = blob

        object_data = {
            "name": "unclassified",
            "weight": 0,
            "xmin": int((x - r + yoffset) * scale),
            "xmax": int((x + r + yoffset) * scale),
            "ymin": int((y - r + xoffset) * scale),
            "ymax": int((y + r + xoffset) * scale)
        }

        return object_data

    objects = [blob_data(blob) for blob in blobs]

    data = {
        "filename": filename,
        "dirname": dirname,
        "width": width,
        "height": height,
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
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.uint8)

        height, width, _ = image.shape

        if width != data['width'] or height != data['height']:
            print("Tamanho da imagem %s diferente do esperado." % filename)
            exit()

        labels =  data['objects']

    return filename, image, labels


def cut_pieces_only_image(img, nx=3, ny=2):
    y_fullsize, x_fullsize, _ = img.shape
    xsize = nx * config.IMG_SIZE
    ysize = ny * config.IMG_SIZE

    yoffset = int((x_fullsize - xsize) / 2)
    xoffset = int((y_fullsize - ysize) / 2)

    img_pieces = []
    for i in range(ny):
        img_line = []
        for j in range(nx):
            x1 = i * config.IMG_SIZE + xoffset
            x2 = (i + 1) * config.IMG_SIZE + xoffset

            y1 = j * config.IMG_SIZE + yoffset
            y2 = (j + 1) * config.IMG_SIZE + yoffset

            img_cut = img[x1:x2, y1:y2]
            img_cut = cv.resize(src=img_cut, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv.INTER_AREA)

            img_line.append(img_cut)

        img_pieces.append(img_line)

    return xoffset, yoffset, img_pieces


def cut_pieces(img, dmap, nx=3, ny=2):
    y_fullsize, x_fullsize, _ = img.shape
    xsize = nx * config.IMG_SIZE
    ysize = ny * config.IMG_SIZE

    xoffset = int((y_fullsize - ysize) / 2)
    yoffset = int((x_fullsize - xsize) / 2)

    img_pieces = []
    map_pieces = []
    for i in range(ny):
        img_line = []
        map_line = []
        for j in range(nx):
            x1 = i * config.IMG_SIZE + xoffset
            y1 = j * config.IMG_SIZE + yoffset
            x2 = (i + 1) * config.IMG_SIZE + xoffset
            y2 = (j + 1) * config.IMG_SIZE + yoffset

            img_cut = img[x1:x2, y1:y2]
            map_cut = dmap[x1:x2, y1:y2]

            img_cut = cv.resize(src=img_cut, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv.INTER_AREA)
            map_cut = cv.resize(src=map_cut, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv.INTER_AREA)

            img_line.append(img_cut)
            map_line.append(map_cut)

        img_pieces.append(img_line)
        map_pieces.append(map_line)

    return img_pieces, map_pieces


def show_pieces(img_pieces, map_pieces):
    nx = len(img_pieces[0])
    ny = len(img_pieces)

    _, axs = plt.subplots(ny, nx, sharey=True)

    for i in range(ny):
        for j in range(nx):
            img_cut = img_pieces[i][j]
            map_cut = map_pieces[i][j]

            axs[i, j].imshow(img_cut)
            axs[i, j].imshow(map_cut, alpha=0.5, cmap="plasma")

    plt.show()
    exit()


def generate_dmap(image, labels):
    im_h, im_w, _ = image.shape
    dmap = np.zeros((im_h, im_w), np.float32)

    for label in labels:
        #if label['name'] == 'normal':
        #    continue

        xmin = label['xmin'] * im_w
        xmax = label['xmax'] * im_w
        ymin = label['ymin'] * im_h
        ymax = label['ymax'] * im_h

        x = int(xmin + (xmax - xmin) / 2)
        y = int(ymin + (ymax - ymin) / 2)

        w = 100 # label['weight'] * 100

        s = (xmax - xmin + ymax - ymin) / 8

        dmap += gaussian_kernel(center=(x, y), map_size=(im_h, im_w), A=w, sx=s, sy=s)
    
    # img_pieces, map_pieces = cut_pieces(img, dmap)
    # show_pieces(img_pieces, map_pieces)

    return dmap
