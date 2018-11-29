import tensorflow as tf

from utils import config

model_id = 'simple_4_4'

initializer = tf.variance_scaling_initializer()

n_layer = 0


def conv2d(x, w, k, s, activation=tf.nn.relu):
    global n_layer
    n_layer += 1
    name = 'CONV' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.conv2d(inputs=x, filters=w, kernel_size=k, strides=s, activation=activation,
                               kernel_initializer=initializer, bias_initializer=initializer, padding='SAME')

    print(name, out.shape)
    return out


def maxpool(x, s, p):
    global n_layer
    name = 'POOL' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.max_pooling2d(
            inputs=x, pool_size=p, strides=s, padding='SAME')

    print(name, out.shape)
    return out


def conv2d_t(x, w, k, s, activation=tf.nn.relu):
    global n_layer
    n_layer += 1
    name = 'CONVT' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.conv2d_transpose(inputs=x, filters=w, kernel_size=k, strides=s, activation=activation,
                                         kernel_initializer=initializer, bias_initializer=initializer, padding='SAME')

    print(name, out.shape)
    return out


def model(x):
    with tf.name_scope('INPUT'):
        x = tf.truediv(tf.cast(x, tf.float32), 255.0)
        print("INPUT " + str(x.shape))

    x = conv2d(x, w=64, k=5, s=1)
    x = maxpool(x, s=2, p=2)

    x = conv2d(x, w=128, k=5, s=1)
    x = maxpool(x, s=2, p=2)

    x = conv2d(x, w=256, k=3, s=1)
    x = maxpool(x, s=2, p=2)

    x = conv2d(x, w=512, k=3, s=1)
    x = maxpool(x, s=2, p=2)

    x = conv2d_t(x, w=512, k=3, s=2)

    x = conv2d_t(x, w=256, k=3, s=2)

    x = conv2d_t(x, w=128, k=3, s=2)

    x = conv2d_t(x, w=64, k=3, s=2)

    x = conv2d(x, w=1, k=3, s=1)

    return x


def map_loss(y_pred, y_true):
    return tf.multiply(tf.losses.mean_squared_error(y_true, y_pred), config.IMG_SIZE * config.IMG_SIZE)


def cnt_loss(count_pred, y_true):
    count_true = tf.reduce_sum(y_true, [1, 2])
    relative_error = tf.abs(tf.truediv(
        tf.subtract(count_true, count_pred), count_true))

    return tf.reduce_mean(relative_error)
