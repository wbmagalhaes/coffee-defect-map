import matplotlib.pyplot as plt
import numpy as np

from utils import density_map

def map_to_2darray(dmap):
    size = len(dmap)
    return dmap.reshape((size, size))

def show_img_dmap_overlay(img, dmap):
    plt.imshow(img)

    dmap = map_to_2darray(dmap)
    plt.imshow(dmap, alpha=.7, cmap="plasma")
    plt.show()

def show_img_result(img, result):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Img')
    ax1.imshow(img)
    
    result = map_to_2darray(result)
    result_int = density_map.integrate(result)
    result_sum = density_map.sum(result)
    ax2 = fig.add_subplot(122)
    ax2.title.set_text('Result: int:{:.2f} - sum:{:.2f}'.format(result_int, result_sum))
    ax2.imshow(result, cmap="plasma")

    plt.show()
    
def show_img_dmap_result(img, dmap, result):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.title.set_text('Img')
    ax1.imshow(img)
    
    result = map_to_2darray(result)
    result_int = density_map.sum(result)
    ax2 = fig.add_subplot(132)
    ax2.title.set_text('Result: {:.2f}'.format(result_int))
    ax2.imshow(result, cmap="plasma")

    dmap = map_to_2darray(dmap)
    dmap_int = density_map.sum(dmap)
    ax3 = fig.add_subplot(133)
    ax3.title.set_text('Expected: {:.0f}'.format(dmap_int))
    ax3.imshow(map_to_2darray(dmap), cmap="plasma")

    plt.show()
