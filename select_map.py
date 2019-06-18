from math import sqrt
from skimage.feature import blob_log

import matplotlib.pyplot as plt
import cv2 as cv

def select_in_map(dmap, min_size=5):
    dmap /= dmap.max()

    blobs_log = blob_log(dmap, min_sigma=5, max_sigma=7, num_sigma=2, threshold=0.05)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    return [blob for blob in blobs_log if blob[2] >= min_size]


def mark_in_img(img, dmap, min_size=5):
    blobs = select_in_map(dmap, min_size)

    for blob in blobs:
        y, x, r = blob

        x = int(x - r)
        y = int(y - r)
        w = int(2 * r)

        cv.rectangle(img, (x, y), (x + w, y + w), (250, 51, 51), 1)

    return blobs