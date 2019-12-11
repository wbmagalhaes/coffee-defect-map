from utils import tfrecords, visualize
from utils.augmentation import rotate, flip, crop, gaussian, clip01

ds = tfrecords.read(['./data/segmentation_test.tfrecord'], im_size=512)

ds = crop(ds, im_size=256)
ds = gaussian(ds, stddev=0.01)
ds = rotate(ds)
ds = flip(ds)
ds = clip01(ds)

ds = ds.shuffle(buffer_size=400).batch(4)

visualize.plot_dataset(ds)
