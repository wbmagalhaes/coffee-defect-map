import os
import glob
import cv2

from CoffeeUNet import create_model

from utils import data_reader, visualize

import matplotlib.pyplot as plt

IMAGE_SIZE = 128
RESULT_ALPHA = 0.4

sample_dirs = [
    'C:/Users/Usuario/Desktop/cafe_imgs/amostras/84A',
    'C:/Users/Usuario/Desktop/cafe_imgs/amostras/248A'
]

model = create_model()
model.load_weights('./results/coffeeunet18.h5')

for _dir in sample_dirs:
    addrs = glob.glob(os.path.join(_dir, '*.jpg'))
    for addr in addrs:
        print(f'Loading image: {addr}')

        original_img = cv2.imread(addr)

        color_image, grey_image = data_reader.prepare_image(original_img, IMAGE_SIZE)
        result = model.predict(grey_image)
        show = visualize.show_combined(color_image, result[0], RESULT_ALPHA, True)

        plt.imshow(show)
        plt.axis('off')
        plt.show()
