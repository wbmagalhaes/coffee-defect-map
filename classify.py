import os
import glob
import cv2

from utils import data_reader, visualize, reload_model

model_name = 'CoffeeUNet18'
epoch = 0

IMAGE_SIZE = 256
RESULT_ALPHA = 0.4

sample_dirs = [
    'C:/Users/Usuario/Desktop/cafe_imgs/amostras/84A',
    'C:/Users/Usuario/Desktop/cafe_imgs/amostras/248A'
]

model = reload_model.from_json(model_name, epoch)

for _dir in sample_dirs:
    addrs = glob.glob(os.path.join(_dir, '*.jpg'))
    for addr in addrs:
        print(f'Loading image: {addr}')

        original_img = cv2.imread(addr)

        color_imgs, grey_imgs = data_reader.prepare_image(original_img, IMAGE_SIZE, 2, 2)
        results = model.predict(grey_imgs)
        show = visualize.show_combined(color_imgs, results, RESULT_ALPHA, 2, 2)

        cv2.imshow("Result", show)
        cv2.waitKey()
