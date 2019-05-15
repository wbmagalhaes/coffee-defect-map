from math import sqrt
from skimage.feature import blob_log

import matplotlib.pyplot as plt


def select_in_map(dmap, min_size=5):
    dmap /= dmap.max()

    shape = dmap.shape
    dmap = dmap.reshape((shape[0], shape[1]))

    blobs_log = blob_log(dmap, min_sigma=5, max_sigma=6,
                         num_sigma=2, threshold=0.05)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_log = [blob for blob in blobs_log if blob[2] >= min_size]

    # _, axs = plt.subplots(nrows=1, ncols=2)
    # axs[0].imshow(image, interpolation=None)
    # axs[1].imshow(dmap, interpolation=None, cmap='jet')

    # for blob in blobs_log:
    #     y, x, r = blob

    #     x = x - r
    #     y = y - r
    #     w = 2 * r

    #     c = plt.Rectangle((x, y), w, w, color='r', linewidth=1, fill=False)
    #     axs[0].add_patch(c)

    # plt.show()

    return blobs_log
