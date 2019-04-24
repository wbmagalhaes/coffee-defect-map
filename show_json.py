import glob
import cv2 as cv

import matplotlib.pyplot as plt

from utils.data_reader import read_json

addrs = glob.glob('C:/Users/Usuario/Desktop/coffee-defect-map/result/*.json')
for addr in addrs:
    filename, img, labels = read_json(addr)

    _, ax = plt.subplots()

    ax.set_title(filename)
    ax.imshow(img, interpolation=None)
    ax.axis('off')

    for label in labels:
        xmin = label['xmin']
        xmax = label['xmax']
        ymin = label['ymin']
        ymax = label['ymax']

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        c = plt.Rectangle((x, y), w, h, color='r', linewidth=1, fill=False)
        ax.add_patch(c)

    plt.show()

