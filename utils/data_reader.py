import os

import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET

from utils import config
from utils import labelmap

def read_xml(addr):
    tree = ET.parse(addr)
    root = tree.getroot()

    xsize = int(root.find('size').find('width').text)
    ysize = int(root.find('size').find('height').text)
    
    filename = root.find('filename').text
    print(filename)

    image = cv.imread(os.path.join(config.IMGS_DIR, filename))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.uint8)
    
    imgs = []
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        
        croped = image[ymin:ymax, xmin:xmax]
        imgs.append(croped)

        xmin *= config.IMG_SIZE / xsize
        xmax *= config.IMG_SIZE / xsize
        ymin *= config.IMG_SIZE / ysize
        ymax *= config.IMG_SIZE / ysize
        
        xpos = (xmax + xmin) / 2
        ypos = (ymax + ymin) / 2
        size = (xmax - xmin + ymax - ymin) / 4

        weight = labelmap.weight_of_label(name)
        
        label = {
            'xpos': xpos,
            'ypos': ypos,
            'size': size,
            'weight': weight
            }
        
        labels.append(label)
    
    image = cv.resize(image, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv.INTER_AREA)
    
    return filename, image, imgs, labels
