from utils import tfrecords, visualize
from utils.augmentation import rotate, flip, crop, gaussian, clip01

ds = tfrecords.read(['./data/segmentation_train.tfrecord'], im_size=512)

ds = crop(ds, im_size=256)
ds = rotate(ds)
ds = flip(ds)
ds = clip01(ds)

ds = ds.batch(4)

visualize.plot_dataset(ds, cmap='gray')
