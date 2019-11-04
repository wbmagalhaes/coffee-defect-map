import os

from utils import data_reader, tfrecords
from random import shuffle

img_dirs = [
    'C:/Users/Usuario/Desktop/cafe_imgs/segmentation_imgs'
]

data_dir = './data'

training_percentage = 0.8

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

train_path = os.path.join(data_dir, 'data_train.tfrecord')
test_path = os.path.join(data_dir, 'data_test.tfrecord')

data = data_reader.load(img_dirs, final_size=256)
shuffle(data)

train_num = int(len(data) * training_percentage)

train_data = data[:train_num]
test_data = data[train_num:]

print(f'{len(train_data)} train images.')
print(f'{len(test_data)} test images.')

print('Writing tfrecords...')
tfrecords.write(train_path, train_data)
tfrecords.write(test_path, test_data)
print('Finished.')
