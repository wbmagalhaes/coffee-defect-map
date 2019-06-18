import tensorflow as tf

from utils import model as cnn
from utils import labelmap

model_id = 'CoffeeUNet18'


def model(x):
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
    x = cnn.crop_and_concat(x, out3)

    x = cnn.conv2d(x, w=256, k=3, s=1)
    x = cnn.conv2d(x, w=256, k=3, s=1)

    x = cnn.conv2d_t(x, w=128, k=2, s=2)
    x = cnn.crop_and_concat(x, out2)

    x = cnn.conv2d(x, w=128, k=3, s=1)
    x = cnn.conv2d(x, w=128, k=3, s=1)

    x = cnn.conv2d_t(x, w=64, k=2, s=2)
    x = cnn.crop_and_concat(x, out1)

    x = cnn.conv2d(x, w=64, k=3, s=1)
    x = cnn.conv2d(x, w=64, k=3, s=1)

    x = cnn.conv2d(x, w=1, k=1, s=1, activation=tf.nn.relu)

    return x
