from utils import tfrecords, other, visualize, reload_model

model_name = 'CoffeeUNet18'
epoch = 0

dataset = tfrecords.read(['./data/segmentation_test.tfrecord'])
dataset = other.resize(dataset).map(other.normalize)

x_data, y_true = zip(*[data for data in dataset])

model = reload_model.from_json(model_name, epoch)

y_pred = model.predict(dataset.batch(32))

visualize.plot_images(x_data[:16], y_true[:16], y_pred[:16])
