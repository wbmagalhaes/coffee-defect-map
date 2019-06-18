import tensorflow as tf

from utils import config
from utils import labelmap

n_layer = 0

kernel_initializer = None  # tf.initializers.he_uniform()
bias_initializer = tf.initializers.zeros()


def conv2d(x, w, k, s, activation=tf.nn.leaky_relu):
    global n_layer
    n_layer += 1
    name = 'CONV' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.conv2d(
            inputs=x,
            filters=w,
            kernel_size=k,
            strides=s,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            padding='SAME',
            name=name)

    print(name, out.shape)
    return out


def conv2d_t(x, w, k, s, activation=tf.nn.leaky_relu):
    global n_layer
    n_layer += 1
    name = 'CONVT' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.conv2d_transpose(
            inputs=x,
            filters=w,
            kernel_size=k,
            strides=s,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            padding='SAME',
            name=name)

    print(name, out.shape)
    return out


def maxpool(x, k, s):
    global n_layer
    name = 'POOL' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=k,
            strides=s,
            padding='SAME',
            name=name)

    print(name, out.shape)
    return out


def crop_and_concat(x1, x2):
    name = 'CONCAT'

    with tf.name_scope(name):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)

        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        x1_crop = tf.slice(x1, offsets, x2_shape)

        out = tf.concat([x1_crop, x2], 3)

    print(name, str(out.shape))
    return out


def dense(x, w, activation=tf.nn.leaky_relu):
    global n_layer
    n_layer += 1
    name = 'DENSE' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.dense(
            inputs=x,
            units=w,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=name)

    print(name, out.shape)
    return out


def map_loss(y_pred, y_true):
    abs_diff = tf.losses.absolute_difference(labels=y_true, predictions=y_pred)
    one_minus = tf.clip_by_value(tf.subtract(1.0, abs_diff), 1e-10, 1.0)
    return -tf.log(one_minus)


def cnt_loss(count_pred, y_true):
    count_true = tf.reduce_sum(y_true, [1, 2])
    relative_error = tf.abs(tf.truediv(tf.subtract(count_true, count_pred), count_true))
    return tf.minimum(0.9999, tf.reduce_mean(relative_error))
