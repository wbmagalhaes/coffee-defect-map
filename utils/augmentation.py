import tensorflow as tf
import numpy as np

from utils import other

import random


def apply(dataset, im_size=256, types=['rotate, flip, crop']):
    def rotate(x, y):
        coin = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        x = tf.image.rot90(x, coin)
        y = tf.image.rot90(y, coin)
        return x, y

    def flip(x, y):
        seed = int(random.random() * 1234.)
        x = tf.image.random_flip_left_right(x, seed=seed)
        y = tf.image.random_flip_left_right(y, seed=seed)

        seed = int(random.random() * 1234.)
        x = tf.image.random_flip_up_down(x, seed=seed)
        y = tf.image.random_flip_up_down(y, seed=seed)
        return x, y

    def crop(x, y):
        seed = int(random.random() * 1234.)
        x = tf.image.random_crop(x, size=[im_size, im_size, 1], seed=seed)
        y = tf.image.random_crop(y, size=[im_size, im_size, 1], seed=seed)
        return x, y

    for t in types:
        if t == 'rotate':
            dataset = dataset.map(rotate, num_parallel_calls=4)
        elif t == 'flip':
            dataset = dataset.map(flip, num_parallel_calls=4)
        elif t == 'crop':
            dataset = dataset.map(crop, num_parallel_calls=4)
        else:
            continue

    dataset = dataset.map(other.clip01, num_parallel_calls=4)
    return dataset
