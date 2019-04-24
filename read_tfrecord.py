import tensorflow as tf

import utils.config as config
from utils.tfrecords import get_data
from utils.data_augment import aug_data
from utils import visualize

from utils import density_map
from select_map import select_in_map

import itertools

data_x, data_y = get_data(filenames=[config.TESTING_PATH], shuffle=True)

x = tf.placeholder(tf.uint8, [None, None, None, 3])
y = tf.placeholder(tf.float32, [None, None, None, 1])

augument_op = aug_data(x, y)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    imgs, dmaps = sess.run(augument_op, feed_dict={x: data_x, y: data_y})

    print(len(imgs))

    defects = []
    for dmap in dmaps:
        size = len(dmap)
        dmap = dmap.reshape((size, size))
        defects.append(density_map.sum(dmap))

    max_d = int(max(defects))
    min_d = int(min(defects))
    med_d = int(sum(defects) / len(defects))

    print("max:", max_d)
    print("min:", min_d)
    print("med:", med_d)

    for img, dmap in itertools.zip_longest(imgs, dmaps):
        select_in_map(img, dmap)
        visualize.show_img_dmap_overlay(img, dmap)

    exit()