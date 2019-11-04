from utils import tfrecords, augmentation, other, visualize

dataset = tfrecords.read(['./data/data_train.tfrecord'])
dataset = dataset.map(other.normalize, num_parallel_calls=4)

dataset = augmentation.apply(dataset)
visualize.plot_dataset(dataset.batch(16))
