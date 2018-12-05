import tensorflow as tf

from utils import config
from utils.tfrecords import get_data
from utils import visualize

model_id = 'CoffeeNet12'
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

    for i in range(len(val_x)):
        visualize.show_img_dmap_result(val_x[i], val_y[i], dmaps[i])

    print('Predictions completed.')
