import tensorflow as tf
import random


def rotate(dataset):
    def apply(x, y):
        coin = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        x = tf.image.rot90(x, coin)
        y = tf.image.rot90(y, coin)
        return x, y

    return dataset.map(apply, num_parallel_calls=4)


def flip(dataset):
    def horizontal(x, y):
        seed = int(random.random() * 1234.)
        x = tf.image.random_flip_left_right(x, seed=seed)
        y = tf.image.random_flip_left_right(y, seed=seed)
        return x, y

    dataset = dataset.map(horizontal, num_parallel_calls=4)

    def vertical(x, y):
        seed = int(random.random() * 1234.)
        x = tf.image.random_flip_up_down(x, seed=seed)
        y = tf.image.random_flip_up_down(y, seed=seed)
        return x, y

    return dataset.map(vertical, num_parallel_calls=4)


def crop(dataset, im_size=256):
    def apply(x, y):
        seed = int(random.random() * 1234.)
        x = tf.image.random_crop(x, size=[im_size, im_size, 1], seed=seed)
        y = tf.image.random_crop(y, size=[im_size, im_size, 1], seed=seed)
        return x, y

    return dataset.map(apply, num_parallel_calls=4)


def gaussian(dataset, stddev=1/255):
    def apply(x, y):
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev, dtype=tf.float32)
        x = x + noise
        return x, y

    return dataset.map(apply, num_parallel_calls=4)


def clip01(dataset):
    def apply(x, y):
        x = tf.clip_by_value(x, 0, 1)
        return x, y

    return dataset.map(apply, num_parallel_calls=4)
