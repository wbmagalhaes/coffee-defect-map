import glob
import numpy as np
import cv2 as cv

from utils import config
from utils.tfrecords import write_tfrecords
from utils.data_reader import read_xml, generate_dmap
from random import shuffle

def create_data(addr):
    _, img, labels = read_xml(addr)

    scale = 8
    img = cv.resize(src=img, dsize=None, fx=1/scale, fy=1/scale, interpolation=cv.INTER_AREA)

    dmap = generate_dmap(img, labels)
    return { 'img': img, 'map': dmap }


addrs = glob.glob(config.IMGS_DIR + '*.xml')
imgs_data = [create_data(addr) for addr in addrs]

shuffle(imgs_data)
img_count = len(imgs_data)
print(img_count, 'Images loaded.')

train_count = int(config.TRAIN_PERCENTAGE * img_count)
val_count = int((img_count - train_count) / 2)

train_data = imgs_data[0:train_count]
test_data = imgs_data[train_count:train_count + val_count]
val_data = imgs_data[train_count + val_count:]

write_tfrecords(config.TRAINING_PATH, train_data)
print('Finished Training Data: {} Images.'.format(len(train_data)))

write_tfrecords(config.TESTING_PATH, test_data)
print('Finished Testing Data: {} Images.'.format(len(test_data)))

write_tfrecords(config.VALIDATION_PATH, val_data)
print('Finished Validation Data: {} Images.'.format(len(val_data)))
