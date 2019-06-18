import tensorflow as tf
import math

import numpy as np

from utils import config
from utils.tfrecords import get_data
from utils import visualize

from utils.data_reader import cut_pieces, show_pieces

from select_map import select_in_map, mark_in_img

model_id = 'CoffeeUNet18'
print('Using model', model_id)

export_dir = 'saved_models/' + model_id + '/'

val_x, val_y = get_data([config.VALIDATION_PATH], shuffle=False)

print('Validation data loaded.')

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(tf.uint8, [None, None, None, 1])
    y = tf.placeholder(tf.float32, [None, None, None, 1])

    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(sess, ["serve"], export_dir)
    graph = tf.get_default_graph()
    print('Graph restored.')

    print('Starting predictions.')
    feed_dict = {'inputs/img_input:0': val_x}
    dmaps, counts = sess.run(['result/dmap:0', 'result/count:0'], feed_dict=feed_dict)

    real_y = np.sum(val_y, axis=(1, 2, 3)) / 100

    pred_y = []
    for i in range(len(val_x)):
        blobs = select_in_map(dmaps[i])
        pred_y.append(len(blobs))

    error = (pred_y - real_y)
    mse = math.sqrt(np.mean(error ** 2))
    print('MSE: {:.2f}'.format(mse))

    abs_error = abs(error)
    mae = np.mean(abs_error)
    print('MAE: {:.2f}'.format(mae))

    rel_error = abs_error / real_y
    mre = np.mean(rel_error)
    print('MRE: {:.2f}%'.format(mre * 100))

    for i in range(len(val_x)):
        img = np.dstack((val_x[i], val_x[i], val_x[i]))

        blobs = mark_in_img(img, dmaps[i])
        visualize.show_img_dmap_result(img, len(blobs), val_y[i], dmaps[i], pred_y[i])

    print('Predictions completed.')
