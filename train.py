import tensorflow as tf
import os
import json

from utils import tfrecords, visualize, losses, metrics
from utils.augmentation import rotate, flip, crop, gaussian, clip01

from CoffeeUNet import create_model

# Load train/test data
train_ds = tfrecords.read(['./data/segmentation_train.tfrecord'], im_size=512)
test_ds = tfrecords.read(['./data/segmentation_test.tfrecord'], im_size=512)
test_ds = crop(test_ds, im_size=256)

# Apply augmentations
train_ds = crop(train_ds, im_size=256)
train_ds = rotate(train_ds)
train_ds = flip(train_ds)
train_ds = clip01(train_ds)

# Set batchs
batch_size = 16
train_ds = train_ds.repeat().shuffle(buffer_size=400).batch(batch_size)
test_ds = test_ds.repeat().shuffle(buffer_size=400).batch(batch_size)

# Plot some images
visualize.plot_dataset(train_ds)

# Define model
model_name = 'CoffeeUNet10'
model = create_model(
    input_shape=(256, 256, 1),
    num_layers=2,
    filters=16,
    num_classes=1,
    kernel_initializer='he_normal',
    kernel_regularizer=tf.keras.regularizers.l2(0.01),
    bias_initializer=tf.keras.initializers.Constant(value=0.1),
    leaky_relu_alpha=0.02,
    output_activation='sigmoid')

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    loss=losses.JaccardDistance(smooth=100),
    metrics=[metrics.IoU(smooth=1.)]
)
model.summary()

# Save model
savedir = os.path.join('results', model_name)
if not os.path.isdir(savedir):
    os.mkdir(savedir)

json_config = model.to_json()
with open(savedir + '/model.json', 'w') as f:
    json.dump(json_config, f)

# Save weights
model.save_weights(savedir + '/epoch-0000.h5')
filepath = savedir + '/epoch-{epoch:04d}.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath,
    save_weights_only=True,
    verbose=1,
    period=1
)

# Tensorboard visualization
logdir = os.path.join('logs', model_name)
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=1,
    write_graph=True,
    profile_batch=0,
    update_freq='epoch'
)

# Training
history = model.fit(
    train_ds,
    steps_per_epoch=20,
    epochs=1,
    verbose=1,
    validation_data=test_ds,
    validation_freq=1,
    validation_steps=5,
    callbacks=[checkpoint, tb_callback]
)
