import glob
import math
import numpy as np
import cv2 as cv

from utils import config
from utils.tfrecords import write_tfrecords
from utils.data_reader import read_xml
from utils.density_map import gaussian_kernel
from utils import visualize
from random import shuffle

IMG_NUM = 2000

COFFEE_SIZE = 16
NUM_COFFEE = int(config.IMG_SIZE / COFFEE_SIZE)

# image = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), np.uint8)
# density_map = np.zeros((config.IMG_SIZE, config.IMG_SIZE), np.float32)
# for i in range(NUM_COFFEE * NUM_COFFEE):
#     xmin = int((i % NUM_COFFEE) * COFFEE_SIZE)
#     ymin = int(math.floor(i / NUM_COFFEE) * COFFEE_SIZE)
#     xmax = xmin + COFFEE_SIZE
#     ymax = ymin + COFFEE_SIZE

#     xpos = (xmax + xmin) / 2
#     ypos = (ymax + ymin) / 2
    
#     density_map += gaussian_kernel((xpos, ypos), config.IMG_SIZE,
#                                        A=10, sx=COFFEE_SIZE/4, sy=COFFEE_SIZE/4)

# visualize.show_img_result(image, density_map)

# exit()

def build_img(data):
    image = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), np.uint8)
    density_map = np.zeros((config.IMG_SIZE, config.IMG_SIZE), np.float32)

    for i in range(len(data)):
        img = data[i]['img']
        label = data[i]['label']

        xmin = int((i % NUM_COFFEE) * COFFEE_SIZE)
        ymin = int(math.floor(i / NUM_COFFEE) * COFFEE_SIZE)
        xmax = xmin + COFFEE_SIZE
        ymax = ymin + COFFEE_SIZE

        xpos = (xmax + xmin) / 2
        ypos = (ymax + ymin) / 2

        image[ymin:ymax, xmin:xmax, :3] = cv.resize(
            img, (COFFEE_SIZE, COFFEE_SIZE), interpolation=cv.INTER_AREA)
        density_map += gaussian_kernel((xpos, ypos), (config.IMG_SIZE, config.IMG_SIZE),
                                       A=label['weight']*100, sx=COFFEE_SIZE/4, sy=COFFEE_SIZE/4)

    return image, density_map


all_data = []
for addr in glob.glob(config.IMGS_DIR + '*.xml'):
    _, _, imgs, labels = read_xml(addr)
    for i in range(len(imgs)):
        data = {'img': imgs[i], 'label': labels[i]}
        all_data.append(data)

print(len(all_data), 'Images loaded.')

imgs_data = []
for i in range(IMG_NUM):
    print(i)
    shuffle(all_data)
    img, dmap = build_img(all_data[0:int(NUM_COFFEE*NUM_COFFEE)])
    imgs_data.append({'img': img, 'map': dmap})

train_count = int(config.TRAIN_PERCENTAGE * IMG_NUM)
val_count = int((IMG_NUM - train_count) / 2)

train_data = imgs_data[0:train_count]
test_data = imgs_data[train_count:train_count + val_count]
val_data = imgs_data[train_count + val_count:]

write_tfrecords(config.TRAINING_PATH, train_data)
print('Finished Training Data: {} Images.'.format(len(train_data)))

write_tfrecords(config.TESTING_PATH, test_data)
print('Finished Testing Data: {} Images.'.format(len(test_data)))

write_tfrecords(config.VALIDATION_PATH, val_data)
print('Finished Validation Data: {} Images.'.format(len(val_data)))
