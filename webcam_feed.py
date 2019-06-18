import tensorflow as tf
import glob
import os

import numpy as np
import cv2 as cv

from utils import config
from utils.data_reader import create_xml, create_json

import matplotlib.pyplot as plt

model_id = 'CoffeeUNet18'
save_path = 'C:/Users/Usuario/Desktop/cafe_imgs/webcam_imgs'

IMAGE_SCALE = 3
SELECTION_SIZE = 3

video = cv.VideoCapture(0)
video.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
video.set(cv.CAP_PROP_FPS, 30)
video.set(cv.CAP_PROP_AUTOFOCUS, False)

font = cv.FONT_HERSHEY_SIMPLEX


def save_image(im, selections):
    n = len(glob.glob(save_path + '/*.jpg'))
    addr = '{}/img_{:03}.jpg'.format(save_path, n)
    print("addr: " + addr)

    cv.imwrite(addr, im)
    create_xml(addr, im, selections)
    create_json(addr, im, selections)


with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()
    tf.saved_model.loader.load(sess, ["serve"], 'saved_models/' + model_id + '/')

    graph = tf.get_default_graph()
    print('Graph restored.')

    input_x = graph.get_tensor_by_name('inputs/img_input:0')

    density_map = graph.get_tensor_by_name('result/dmap:0')
    local_maxima = graph.get_tensor_by_name('result/maxima:0')

    while(1):
        _, original_img = video.read()
        img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY).astype(np.float32)
        img = cv.resize(src=img, dsize=None, fx=1/IMAGE_SCALE, fy=1/IMAGE_SCALE, interpolation=cv.INTER_AREA)
        img /= 255

        h, w = img.shape
        img_input = np.reshape(img, (1, h, w, 1))

        dmap, maxima = sess.run([density_map, local_maxima],feed_dict={input_x: img_input})

        coords = np.argwhere(maxima[0] > 0.25)

        im = original_img.copy()

        def coord_data(coord):
            y, x, _ = coord

            w = SELECTION_SIZE / dmap[0][y][x][0]

            xmin = int((x - w) * IMAGE_SCALE)
            xmax = int((x + w) * IMAGE_SCALE)
            ymin = int((y - w) * IMAGE_SCALE)
            ymax = int((y + w) * IMAGE_SCALE)

            cv.drawMarker(im, (x * IMAGE_SCALE, y * IMAGE_SCALE), (51, 51, 250), cv.MARKER_CROSS, 16)
            cv.rectangle(im, (xmin, ymin), (xmax, ymax), (51, 51, 250), 1)

            return [xmin, ymin, xmax, ymax]

        selections = [coord_data(coord) for coord in coords]

        cv.putText(im, 'Selected: {}'.format(len(selections)), (10, 710), font, 1, (51, 51, 250), 2, cv.LINE_AA)
        cv.imshow("Result", im)

        k = cv.waitKey(30) & 0xff
        if k == 32:  # SPACEBAR pressed
            save_image(original_img, selections)
        elif k == 27:  # ESCAPE pressed
            break

video.release()
cv.destroyAllWindows()
