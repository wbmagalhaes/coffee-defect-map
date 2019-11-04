from CoffeeUNet import create_model

from utils import tfrecords, other, visualize

dataset = tfrecords.read(['./data/data_test.tfrecord'])
dataset = dataset.map(other.resize).map(other.normalize)

visualize.plot_dataset(dataset.batch(2))

x_data, y_true = zip(*[data for data in dataset])

model = create_model()
model.load_weights('./results/coffeeunet18.h5')

y_pred = model.predict(dataset.batch(32))

visualize.plot_images(x_data[:8], y_true[:8], y_pred[:8])
