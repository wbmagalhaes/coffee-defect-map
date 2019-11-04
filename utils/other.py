import tensorflow as tf


def normalize(x, y):
    x = tf.divide(x, 255.)
    return x, y


def clip01(x, y):
    x = tf.clip_by_value(x, 0, 1)
    return x, y


def resize(x, y):
    x = tf.image.resize(x, [128, 128])
    y = tf.image.resize(y, [128, 128])
    return x, y
