import tensorflow as tf
import math

import numpy as np

from utils import config
from utils.tfrecords import get_data
from utils import visualize

model_id = 'CoffeeUNet'
print('Using model', model_id)

export_dir = 'saved_models/' + model_id + '/'

val_x, val_y = get_data([config.VALIDATION_PATH], shuffle=False)
print('Validation data loaded.')

with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(sess, ["serve"], export_dir)

    graph = tf.get_default_graph()
    print('Graph restored.')

    print('Starting predictions.')
    feed_dict = {
        'inputs/image_input:0': val_x  # ,
        # 'inputs/is_training:0': False
    }
    dmaps, counts = sess.run(
        ['result/dmap:0', 'result/count:0'], feed_dict=feed_dict)

    pred_y = np.sum(counts, axis=1)
    real_y = np.sum(val_y, axis=(1, 2, 3))

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
        visualize.show_img_dmap_result(val_x[i], val_y[i], dmaps[i])

    print('Predictions completed.')
