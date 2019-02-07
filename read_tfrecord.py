import tensorflow as tf

import utils.config as config
from utils.tfrecords import get_data
from utils import visualize

from utils import density_map

import itertools

imgs, dmaps = get_data(filenames=[config.TESTING_PATH], shuffle=True)

print(len(imgs))

defects = []
for dmap in dmaps:
    size = len(dmap)
    dmap = dmap.reshape((size, size))
    defects.append(density_map.sum(dmap))

max_d = int(max(defects))
min_d = int(min(defects))
med_d = int(sum(defects) / len(defects))

print("max:", max_d)
print("min:", min_d)
print("med:", med_d)

for img, dmap in itertools.zip_longest(imgs, dmaps):
    visualize.show_img_result(img, dmap)
