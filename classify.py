import tensorflow as tf
import glob
import os

import numpy as np
import cv2 as cv
import math

from utils import config
from utils import visualize

from utils.density_map import gaussian_kernel
from utils.data_reader import read_xml, create_json, cut_pieces_only_image, show_pieces

from select_map import select_in_map

model_id = 'CoffeeUNet18_newimages'
print('Using model', model_id)

export_dir = 'saved_models/' + model_id + '/'

selections = []
reals = []

with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(sess, ["serve"], export_dir)

    graph = tf.get_default_graph()
    print('Graph restored.')

    print('Starting predictions...')

    addrs = glob.glob(
        'C:/Users/Usuario/Desktop/coffee-defect-map/result/*.jpg')
    for addr in addrs:
        print('================')

        _, _, labels = read_xml(addr[:-3] + "xml")
        real_y = len(labels)

        filename = os.path.basename(addr)
        img = cv.imread(addr)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.uint8)

        scale = 8
        img = cv.resize(src=img, dsize=None, fx=1/scale,
                        fy=1/scale, interpolation=cv.INTER_AREA)

        xoffset, yoffset, img_lines = cut_pieces_only_image(img)
        nx = len(img_lines[0])
        ny = len(img_lines)

        img_pieces = np.reshape(
            img_lines, (nx * ny, config.IMG_SIZE, config.IMG_SIZE, 3))

        dmaps, counts = sess.run(
            ['result/dmap:0', 'result/count:0'],
            feed_dict={'inputs/image_input:0': img_pieces})

        img_line1 = np.concatenate((img_pieces[:nx]), axis=1)
        img_line2 = np.concatenate((img_pieces[nx:]), axis=1)
        fullimg = np.concatenate((img_line1, img_line2), axis=0)

        dmaps = np.reshape(dmaps, (nx * ny, config.IMG_SIZE, config.IMG_SIZE))

        map_line1 = np.concatenate((dmaps[:nx]), axis=1)
        map_line2 = np.concatenate((dmaps[nx:]), axis=1)
        fullmap = np.concatenate((map_line1, map_line2), axis=0)

        blobs = select_in_map(fullimg, fullmap)
        create_json(addr, blobs, xoffset, yoffset, scale)

        # visualize.show_selection_dmap(fullimg, fullmap, blobs, real_y)

        pred_y = len(blobs)

        selections.append(pred_y)
        reals.append(real_y)

        print('Grãos: {}'.format(real_y))
        print('Selecionados: {}'.format(pred_y))

        error = abs(pred_y - real_y) / real_y
        print('Erro: {:.2f}%'.format(error * 100))

        print('================')

    print('Predictions completed.')


def errors(p, r):
    p = np.array(p)
    r = np.array(r)

    error = (p - r)
    mse = math.sqrt(np.mean(error ** 2))
    print('MSE: {:.2f}'.format(mse))

    abs_error = abs(error)
    mae = np.mean(abs_error)
    print('MAE: {:.2f}'.format(mae))

    rel_error = abs_error / r
    mre = np.mean(rel_error)
    print('MRE: {:.2f}%'.format(mre * 100))

errors(selections, reals)
