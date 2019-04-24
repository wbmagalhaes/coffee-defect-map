import tensorflow as tf
import math

import numpy as np

from utils import config
from utils.tfrecords import get_data
from utils.data_augment import aug_data
from utils import visualize

from utils.data_reader import cut_pieces, show_pieces

from select_map import select_in_map

model_id = 'CoffeeUNet18_newimages'
print('Using model', model_id)

export_dir = 'saved_models/' + model_id + '/'

val_x, val_y = get_data([config.VALIDATION_PATH], shuffle=False)
# img_pieces, map_pieces = cut_pieces(val_x[0], val_y[0])
# show_pieces(img_pieces, map_pieces)

print('Validation data loaded.')

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(tf.uint8, [None, None, None, 3])
    y = tf.placeholder(tf.float32, [None, None, None, 1])

    augument_op = aug_data(x, y)

    tf.global_variables_initializer().run()

    imgs, real_dmaps = sess.run(augument_op, feed_dict={x: val_x, y: val_y})

    tf.saved_model.loader.load(sess, ["serve"], export_dir)

    graph = tf.get_default_graph()
    print('Graph restored.')

    print('Starting predictions.')
    feed_dict = {
        'inputs/image_input:0': imgs  # ,
        # 'inputs/is_training:0': False
    }
    dmaps, counts = sess.run(
        ['result/dmap:0', 'result/count:0'], feed_dict=feed_dict)

    pred_y = np.sum(counts, axis=1) / 100
    real_y = np.sum(real_dmaps, axis=(1, 2, 3)) / 100

    error = (pred_y - real_y)
    mse = math.sqrt(np.mean(error ** 2))
    print('MSE: {:.2f}'.format(mse))

    abs_error = abs(error)
    mae = np.mean(abs_error)
    print('MAE: {:.2f}'.format(mae))

    rel_error = abs_error / real_y
    mre = np.mean(rel_error)
    print('MRE: {:.2f}%'.format(mre * 100))

    for i in range(len(imgs)):
        select_in_map(imgs[i], dmaps[i])
        visualize.show_img_dmap_result(imgs[i], real_dmaps[i], dmaps[i])

    print('Predictions completed.')
