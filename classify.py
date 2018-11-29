import tensorflow as tf
import glob

import numpy as np
import cv2 as cv

from utils import config
from utils import visualize

export_dir = 'saved_models/simple_4_4/'

imgs = []
for addr in glob.glob('result/*.jpg'):
    image = cv.imread(addr)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.uint8)
    image = cv.resize(image, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv.INTER_AREA)
    imgs.append(image)

print('{} images loaded.'.format(len(imgs)))

with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(sess, ["serve"], export_dir)
    
    graph = tf.get_default_graph()
    print('Graph restored.')
    
    print('Starting predictions.')
    feed_dict = {
        'inputs/image_input:0': imgs#,
        #'inputs/is_training:0': False
    }
    
    dmaps, counts = sess.run(['result/dmap:0', 'result/count:0'], feed_dict=feed_dict)
    
    for i in range(len(imgs)):
        visualize.show_img_result(imgs[i], dmaps[i])
