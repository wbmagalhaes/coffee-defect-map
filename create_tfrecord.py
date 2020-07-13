import os

from utils import data_reader, tfrecords
from random import shuffle

img_dirs = [
    'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs',
    'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs/classificados/ardido',
    'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs/classificados/brocado',
    # 'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs/classificados/chocho',
    # 'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs/classificados/coco',
    # 'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs/classificados/concha',
    'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs/classificados/marinheiro',
    'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs/classificados/normal',
    'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs/classificados/preto',
    # 'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs/classificados/quebrado',
    'E:/William/Documents/Mestrado/cafe_imgs/segmentation_imgs/classificados/verde'
]

data_dir = './data'

training_percentage = 0.8

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

train_path1 = os.path.join(data_dir, 'segmentation_train1.tfrecord')
train_path2 = os.path.join(data_dir, 'segmentation_train2.tfrecord')
train_path3 = os.path.join(data_dir, 'segmentation_train3.tfrecord')
test_path = os.path.join(data_dir, 'segmentation_test.tfrecord')

data = data_reader.load_json(img_dirs, final_size=512)
shuffle(data)

print(f'{len(data)} total images.')
train_num = int(len(data) * training_percentage)

train_data = data[:train_num]
test_data = data[train_num:]

print(f'{len(train_data)} train images.')
print(f'{len(test_data)} test images.')

n = train_num//3
train_data1 = train_data[:n]
train_data2 = train_data[n:2*n]
train_data3 = train_data[2*n:]

print(len(train_data1))
print(len(train_data2))
print(len(train_data3))

print('Writing tfrecords...')
tfrecords.write(train_path1, train_data1)
tfrecords.write(train_path2, train_data2)
tfrecords.write(train_path3, train_data3)
tfrecords.write(test_path, test_data)
print('Finished.')
