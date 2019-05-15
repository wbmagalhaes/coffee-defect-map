import tensorflow as tf

import numpy as np
import cv2 as cv

from utils import config
from utils.data_reader import cut_pieces_only_image

from select_map import select_in_map

model_id = 'CoffeeUNet18_newimages'
print('Using model', model_id)

video = cv.VideoCapture(0)
video.set(cv.CAP_PROP_FRAME_WIDTH, 1600)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 900)
video.set(cv.CAP_PROP_FPS, 30)
video.set(cv.CAP_PROP_AUTOFOCUS, True)

font = cv.FONT_HERSHEY_SIMPLEX

with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(
        sess, ["serve"], 'saved_models/' + model_id + '/')

    graph = tf.get_default_graph()
    print('Graph restored.')

    print('Starting predictions...')

    while(1):
        _, original_img = video.read()

        img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB).astype(np.uint8)

        scale = 3
        img = cv.resize(src=img, dsize=None, fx=1/scale,
                        fy=1/scale, interpolation=cv.INTER_AREA)

        xoffset, yoffset, img_lines = cut_pieces_only_image(img, nx=2, ny=1)
        nx = len(img_lines[0])
        ny = len(img_lines)

        img_pieces = np.reshape(img_lines, (nx * ny, config.IMG_SIZE, config.IMG_SIZE, 3))

        dmaps, counts = sess.run(
            ['result/dmap:0', 'result/count:0'],
            feed_dict={'inputs/image_input:0': img_pieces})

        dmaps = np.reshape(dmaps, (nx * ny, config.IMG_SIZE, config.IMG_SIZE))
        count = sum(counts / 1000)

        fullmap = np.concatenate((dmaps[:nx]), axis=1)
        #map_line2 = np.concatenate((dmaps[nx:]), axis=1)
        #fullmap = np.concatenate((map_line1, map_line2), axis=0)

        blobs = select_in_map(fullmap, min_size=1)
        selected = len(blobs)

        for i in range(nx):
            for j in range(ny):

                x1 = (i * config.IMG_SIZE + yoffset) * scale
                x2 = ((i + 1) * config.IMG_SIZE + yoffset) * scale

                y1 = (j * config.IMG_SIZE + xoffset) * scale
                y2 = ((j + 1) * config.IMG_SIZE + xoffset) * scale

                cv.rectangle(original_img, (x1, y1), (x2, y2), (150, 255, 150), 1)

        for blob in blobs:
            y, x, r = blob

            x = int(x - r + yoffset) * scale
            y = int(y - r + xoffset) * scale
            w = int(2 * r) * scale

            cv.rectangle(original_img, (x, y), (x + w, y + w), (0, 0, 255), 2)

        cv.putText(original_img, 'Selected: %i' % selected, (5, 880), font, 1, (51, 51, 250), 2, cv.LINE_AA)

        cv.imshow("Result", original_img)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

video.release()
cv.destroyAllWindows()
