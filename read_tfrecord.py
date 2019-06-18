import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import itertools

from utils.tfrecords import get_data
import utils.config as config

data_x, data_y = get_data(filenames=[config.TESTING_PATH], shuffle=True)

x = tf.placeholder(tf.float32, [None, None, None, 1])
y = tf.placeholder(tf.float32, [None, None, None, 1])

img = tf.squeeze(x)
dmap = tf.squeeze(y)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    imgs, dmaps = sess.run([img, dmap], feed_dict={x: data_x, y: data_y})

    for img, dmap in itertools.zip_longest(imgs, dmaps):
        count = int(np.sum(dmap))

        plt.imshow(img, cmap="gray")
        plt.imshow(dmap, alpha=.5, cmap="jet")
        plt.title('Count: %i' % (count / 100))
        plt.show()
