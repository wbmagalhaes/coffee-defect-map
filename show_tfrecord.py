from utils import tfrecords, augmentation, other, visualize

dataset = tfrecords.read(['./data/segmentation_test.tfrecord'], im_size=512)
dataset = dataset.map(other.normalize, num_parallel_calls=4)

dataset = augmentation.apply(dataset, types=['crop', 'rotate', 'flip'])
visualize.plot_dataset(dataset.batch(4))
