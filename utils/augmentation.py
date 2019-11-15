import tensorflow as tf
import numpy as np

from utils import other

import random


def apply(dataset, im_size=256):
    def rotate(x, y):
        seed = int(random.random() * 1234.)
        coin = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32, seed=seed)
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

    augmentations = [crop, flip, rotate]
    for f in augmentations:
        dataset = dataset.map(f, num_parallel_calls=4)

    dataset = dataset.map(other.clip01, num_parallel_calls=4)
    return dataset
