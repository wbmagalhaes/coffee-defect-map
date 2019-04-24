import tensorflow as tf
import numpy as np

import utils.config as config


def aug_data(images, labels):
    with tf.name_scope('augument'):
        batch_size = tf.shape(images)[0]

        with tf.name_scope('random_crop'):
            seed = np.random.randint(1234)
            images = tf.map_fn(lambda i: tf.random_crop(
                i, size=[config.IMG_SIZE, config.IMG_SIZE, 3], seed=seed), images)
            labels = tf.map_fn(lambda i: tf.random_crop(
                i, size=[config.IMG_SIZE, config.IMG_SIZE, 1], seed=seed), labels)

        with tf.name_scope('horizontal_flip'):
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            images = tf.where(coin, tf.image.flip_left_right(images), images)
            labels = tf.where(coin, tf.image.flip_left_right(labels), labels)

        with tf.name_scope('vertical_flip'):
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            images = tf.where(coin, tf.image.flip_up_down(images), images)
            labels = tf.where(coin, tf.image.flip_up_down(labels), labels)

        with tf.name_scope('rotate'):
            angle = tf.cast(tf.random_uniform([], 0, 4.0), tf.int32)
            images = tf.image.rot90(images, angle)
            labels = tf.image.rot90(labels, angle)

        with tf.name_scope('gaussian_noise'):
            images = tf.truediv(tf.cast(images, tf.float32), 255.0)
            noise = tf.random_normal(
                shape=[batch_size, config.IMG_SIZE, config.IMG_SIZE, 3],
                mean=0.0,
                stddev=0.02,
                dtype=tf.float32
            )
            # images = tf.clip_by_value(tf.add(images, noise), 0.0, 1.0)

    return images, labels
