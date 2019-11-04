import tensorflow as tf

import random


def normalize(x, y):
    x = tf.divide(x, 255.)
    return x, y


def clip01(x, y):
    x = tf.clip_by_value(x, 0, 1)
    return x, y


def resize(x, y):
    seed = int(random.random() * 1234)
    x = tf.image.random_crop(x, size=[128, 128, 1], seed=seed)
    y = tf.image.random_crop(y, size=[128, 128, 1], seed=seed)

    return x, y
