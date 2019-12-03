import tensorflow as tf

import random


def normalize(x, y):
    x = tf.divide(x, 255.)
    return x, y


def clip01(x, y):
    x = tf.clip_by_value(x, 0, 1)
    return x, y


def resize(dataset, im_size=256):
    def random_crop(x, y):
        seed = int(random.random() * 1234)
        x = tf.image.random_crop(x, size=[im_size, im_size, 1], seed=seed)
        y = tf.image.random_crop(y, size=[im_size, im_size, 1], seed=seed)
        return x, y

    dataset = dataset.map(random_crop, num_parallel_calls=4)
    return dataset
