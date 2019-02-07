import tensorflow as tf

from utils import config
from utils import labelmap

n_layer = 0

initializer = tf.initializers.he_uniform()


def conv2d(x, w, k, s, activation=tf.nn.relu):
    global n_layer
    n_layer += 1
    name = 'CONV' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.conv2d(
            inputs=x, filters=w, kernel_size=k, strides=s, activation=activation,
            kernel_initializer=initializer, bias_initializer=initializer, padding='SAME', name=name)

    print(name, out.shape)
    return out


def conv2d_t(x, w, k, s, activation=tf.nn.relu):
    global n_layer
    n_layer += 1
    name = 'CONVT' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.conv2d_transpose(
            inputs=x, filters=w, kernel_size=k, strides=s, activation=activation,
            kernel_initializer=initializer, bias_initializer=initializer, padding='SAME', name=name)

    print(name, out.shape)
    return out


def maxpool(x, k, s):
    global n_layer
    name = 'POOL' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.max_pooling2d(
            inputs=x, pool_size=k, strides=s, padding='SAME', name=name)

    print(name, out.shape)
    return out


def dense(x, w, activation=tf.nn.relu):
    global n_layer
    n_layer += 1
    name = 'DENSE' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.dense(
            inputs=x, units=w, activation=activation,
            kernel_initializer=initializer, bias_initializer=initializer, name=name)

    print(name, out.shape)
    return out


def map_loss(y_pred, y_true):
    return tf.losses.absolute_difference(labels=y_true, predictions=y_pred)#, weights=config.IMG_SIZE * config.IMG_SIZE)


def cnt_loss(count_pred, y_true):
    count_true = tf.reduce_sum(y_true, [1, 2])
    relative_error = tf.abs(tf.truediv(
        tf.subtract(count_true, count_pred), count_true))

    return tf.reduce_mean(relative_error)
