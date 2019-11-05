import cv2

from CoffeeUNet import create_model

from utils import data_reader, visualize

IMAGE_SIZE = 256
RESULT_ALPHA = 0.4

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video.set(cv2.CAP_PROP_FPS, 30)
video.set(cv2.CAP_PROP_AUTOFOCUS, False)

model = create_model()
model.load_weights('./results/coffeeunet18.h5')

while(1):
    _, original_img = video.read()

    color_imgs, grey_imgs = data_reader.prepare_image(original_img, IMAGE_SIZE, 2, 2)
    results = model.predict(grey_imgs)
    show = visualize.show_combined(color_imgs, results, RESULT_ALPHA, 2, 2)

    cv2.imshow("Result", show)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESCAPE pressed
        break

video.release()
cv2.destroyAllWindows()
