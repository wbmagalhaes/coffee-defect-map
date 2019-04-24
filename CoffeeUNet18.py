import tensorflow as tf

from utils import model as cnn
from utils import labelmap

model_id = 'CoffeeUNet18_newimages'


def model(x):
    with tf.name_scope('INPUT'):
        x = tf.truediv(tf.cast(x, tf.float32), 255.0)
        x = tf.map_fn(lambda i: tf.image.per_image_standardization(i), x)
        # x = tf.image.rgb_to_yuv(x)
        print("INPUT " + str(x.shape))

    x = cnn.conv2d(x, w=64, k=3, s=1)
    out1 = cnn.conv2d(x, w=64, k=3, s=1)
    x = cnn.maxpool(out1, k=2, s=2)

    x = cnn.conv2d(x, w=128, k=3, s=1)
    out2 = cnn.conv2d(x, w=128, k=3, s=1)
    x = cnn.maxpool(out2, k=2, s=2)

    x = cnn.conv2d(x, w=256, k=3, s=1)
    out3 = cnn.conv2d(x, w=256, k=3, s=1)
    x = cnn.maxpool(out3, k=2, s=2)

    x = cnn.conv2d(x, w=512, k=3, s=1)
    x = cnn.conv2d(x, w=512, k=3, s=1)

    x = cnn.conv2d_t(x, w=256, k=2, s=2)
    x = tf.concat([out3, x], 3)
    print("CONCAT", str(x.shape))

    x = cnn.conv2d(x, w=256, k=3, s=1)
    x = cnn.conv2d(x, w=256, k=3, s=1)

    x = cnn.conv2d_t(x, w=128, k=2, s=2)
    x = tf.concat([out2, x], 3)
    print("CONCAT", str(x.shape))

    x = cnn.conv2d(x, w=128, k=3, s=1)
    x = cnn.conv2d(x, w=128, k=3, s=1)

    x = cnn.conv2d_t(x, w=64, k=2, s=2)
    x = tf.concat([out1, x], 3)
    print("CONCAT", str(x.shape))

    x = cnn.conv2d(x, w=64, k=3, s=1)
    x = cnn.conv2d(x, w=64, k=3, s=1)

    x = cnn.conv2d(x, w=1, k=1, s=1)

    return x
