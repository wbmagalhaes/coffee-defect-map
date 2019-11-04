from CoffeeUNet import create_model

from utils import tfrecords, other, visualize

dataset = tfrecords.read(['./data/data_test.tfrecord'])
dataset = dataset.map(other.resize).map(other.normalize)

x_data, y_true = zip(*[data for data in dataset])

model = create_model()
model.load_weights('./results/coffeeunet18.h5')

y_pred = model.predict(dataset.batch(32))

visualize.plot_images(x_data[:16], y_true[:16], y_pred[:16])
