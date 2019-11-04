import cv2

import numpy as np

from CoffeeUNet import create_model

from utils import data_reader

IMAGE_SIZE = 128
RESULT_ALPHA = 0.4

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video.set(cv2.CAP_PROP_FPS, 30)
video.set(cv2.CAP_PROP_AUTOFOCUS, False)

font = cv2.FONT_HERSHEY_SIMPLEX

model = create_model()
model.load_weights('./results/coffeeunet18.h5')

while(1):
    _, original_img = video.read()

    color_image = data_reader.cut_image(original_img, channels=3)
    color_image = color_image.astype(np.float32) / 255.

    grey_image = data_reader.prepare_image(color_image, final_size=IMAGE_SIZE)
    grey_image = cv2.cvtColor(grey_image, cv2.COLOR_BGR2GRAY)
    grey_image = np.reshape(grey_image, (1, IMAGE_SIZE, IMAGE_SIZE, 1))

    result = model.predict(grey_image)
    result = np.squeeze(result[0])
    result = np.stack([np.zeros_like(result), np.zeros_like(result), result], axis=-1)
    result = cv2.resize(src=result, dsize=(color_image.shape[0], color_image.shape[1]))

    show = np.hstack([color_image, result])
    cv2.imshow("Result", show)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESCAPE pressed
        break

video.release()
cv2.destroyAllWindows()
