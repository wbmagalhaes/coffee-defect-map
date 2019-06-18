import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import normalize

from utils import density_map


def show_img_dmap_overlay(img, dmap):
    plt.imshow(img)
    plt.imshow(dmap, alpha=.7, cmap="plasma")
    plt.show()


def show_img_result(img, result):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Img')

    print(np.max(img))
    print(np.min(img))
    #img = normalize(img)
    ax1.imshow(img)

    result_int = density_map.integrate(result)
    result_sum = density_map.sum(result)
    ax2 = fig.add_subplot(122)
    ax2.title.set_text(
        'Result: int:{:.1f} - sum:{:.1f}'.format(result_int, result_sum))
    ax2.imshow(result, cmap="plasma")

    plt.show()


def show_img_dmap_result(img, blobs, real_dmap, result, count):
    img = np.squeeze(img)
    real_dmap = np.squeeze(real_dmap)
    result = np.squeeze(result)

    dmap_int = int(density_map.sum(real_dmap) / 100)

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.title.set_text('Selected: %i/%i' % (blobs, dmap_int))
    ax1.imshow(img, interpolation=None)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ax2 = fig.add_subplot(223)
    # ax2.title.set_text('Count: %.1f' % count)
    # ax2.imshow(result, cmap="jet", interpolation=None)
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    # ax3 = fig.add_subplot(224)
    # ax3.title.set_text('Expected: %i' % dmap_int)
    # ax3.imshow(real_dmap, cmap="jet", interpolation=None)
    # ax3.set_xticks([])
    # ax3.set_yticks([])

    plt.show()


def show_selection_dmap(img, dmap, blobs, real_y=-1):
    _, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].title.set_text('Selected: {}'.format(len(blobs)))
    axs[0].imshow(img, interpolation=None)

    for blob in blobs:
        y, x, r = blob

        x = x - r
        y = y - r
        w = 2 * r

        c = plt.Rectangle((x, y), w, w, color='r', linewidth=1, fill=False)
        axs[0].add_patch(c)

    if real_y >= 0:
        axs[1].title.set_text('Real Count: {}'.format(real_y))
    axs[1].imshow(dmap, interpolation=None, cmap="jet")

    plt.show()