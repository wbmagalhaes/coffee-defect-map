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


train_path = os.path.join(data_dir, 'segmentation_train')
teste_path = os.path.join(data_dir, 'segmentation_teste')
valid_path = os.path.join(data_dir, 'segmentation_valid')

data = data_reader.load_json(img_dirs, final_size=512)
shuffle(data)

print(f'{len(data)} total images.')
train_num = int(len(data) * training_percentage)
teste_num = int(len(data) * (1 - training_percentage)) // 2

train_data = data[:train_num]
teste_data = data[train_num:train_num + teste_num]
valid_data = data[train_num + teste_num:]

print(f'{len(train_data)} train images.')
print(f'{len(teste_data)} teste images.')
print(f'{len(valid_data)} valid images.')


def split_data(path, data, num):
    size = len(data) // num
    for i in range(0, num + 1):
        splitted = data[size * i:size * (i + 1)]
        if len(splitted) > 0:
            tfrecords.write(f"{path}_{i}.tfrecord", splitted)


print('Writing tfrecords...')
split_data(train_path, train_data, 4)
split_data(teste_path, teste_data, 2)
split_data(valid_path, valid_data, 2)
print('Finished.')
