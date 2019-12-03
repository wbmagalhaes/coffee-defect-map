import tensorflow as tf
import numpy as np

from utils import other

import random


def apply(dataset, im_size=256, stddev=0.04):
    def rotate(x, y):
        coin = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        x = tf.image.rot90(x, coin)
        y = tf.image.rot90(y, coin)
        return x, y

    def flip_h(x, y):
        seed = int(random.random() * 1234.)
        x = tf.image.random_flip_left_right(x, seed=seed)
        y = tf.image.random_flip_left_right(y, seed=seed)
        return x, y

    def flip_v(x, y):
        seed = int(random.random() * 1234.)
        x = tf.image.random_flip_up_down(x, seed=seed)
        y = tf.image.random_flip_up_down(y, seed=seed)
        return x, y

    def crop(x, y):
        seed = int(random.random() * 1234.)
        x = tf.image.random_crop(x, size=[im_size, im_size, 1], seed=seed)
        y = tf.image.random_crop(y, size=[im_size, im_size, 1], seed=seed)
        return x, y

    def gaussian(x, y):
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev, dtype=tf.float32)
        x = x + noise * (1.0 - y)
        return x, y

    types = [crop, rotate, flip_h, flip_v, gaussian]
    for t in types:
        dataset = dataset.map(t, num_parallel_calls=4)

    dataset = dataset.map(other.clip01, num_parallel_calls=4)
    return dataset
