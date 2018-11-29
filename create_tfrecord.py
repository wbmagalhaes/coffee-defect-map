import glob
import numpy as np

from utils import config
from utils.tfrecords import write_tfrecords
from utils.data_reader import read_xml
from utils.density_map import gaussian_kernel
from random import shuffle

imgs_data = []
for addr in glob.glob(config.IMGS_DIR + '*.xml'):
    _, img, _, labels = read_xml(addr)
    
    dmap = np.zeros((config.IMG_SIZE, config.IMG_SIZE), np.float32)
    for label in labels:
        dmap += gaussian_kernel((label['xpos'], label['ypos']), config.IMG_SIZE, A=label['weight'],sx=label['size'],sy=label['size'])

    imgs_data.append({'img': img, 'map': dmap })

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
