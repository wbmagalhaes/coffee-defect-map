import tensorflow as tf

from utils import model as cnn

model_id = 'Simple9'


def model(x):
    with tf.name_scope('INPUT'):
        x = tf.truediv(tf.cast(x, tf.float32), 255.0)
        print("INPUT " + str(x.shape))

    x = cnn.conv2d(x, w=64, k=5, s=1)
    x = cnn.maxpool(x, k=2, s=2)

    x = cnn.conv2d(x, w=128, k=5, s=1)
    x = cnn.maxpool(x, k=2, s=2)

    x = cnn.conv2d(x, w=256, k=3, s=1)
    x = cnn.maxpool(x, k=2, s=2)

    x = cnn.conv2d(x, w=512, k=3, s=1)
    x = cnn.maxpool(x, k=2, s=2)

    x = cnn.conv2d_t(x, w=512, k=3, s=2)

    x = cnn.conv2d_t(x, w=256, k=3, s=2)

    x = cnn.conv2d_t(x, w=128, k=3, s=2)

    x = cnn.conv2d_t(x, w=64, k=3, s=2)

    x = cnn.conv2d(x, w=1, k=3, s=1)

    return x
