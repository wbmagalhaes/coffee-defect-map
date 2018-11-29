import tensorflow as tf

import utils.config as config
from utils.tfrecords import get_data
from utils import visualize

imgs, dmaps = get_data(filenames=[config.TESTING_PATH], shuffle=True)

print(len(imgs))
for i in range(len(imgs)):
    visualize.show_img_result(imgs[i], dmaps[i])
