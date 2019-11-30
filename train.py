import tensorflow as tf
import os
import json

from utils import tfrecords, augmentation, other, visualize, losses, metrics
from CoffeeUNet import create_model

# Load train data
train_dataset = tfrecords.read(['./data/segmentation_train.tfrecord'])
train_dataset = train_dataset.map(other.normalize, num_parallel_calls=4)

# Load test data
test_dataset = tfrecords.read(['./data/segmentation_test.tfrecord'])
test_dataset = test_dataset.map(other.normalize, num_parallel_calls=4)

# Apply augmentations
train_dataset = augmentation.apply(train_dataset, types=['crop', 'rotate', 'flip'])
test_dataset = augmentation.apply(test_dataset, types=['crop', 'rotate', 'flip'])

# Set batchs
train_dataset = train_dataset.repeat().shuffle(buffer_size=400).batch(16)
test_dataset = test_dataset.repeat().shuffle(buffer_size=400).batch(16)

# Plot some images
# visualize.plot_dataset(train_dataset)

# Define model
model_name = 'CoffeeUNet18'
model = create_model()
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
model.save_weights(savedir + '/epoch-000.ckpt')
filepath = savedir + '/epoch-{epoch:03d}.ckpt'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, verbose=1, period=50)

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
    train_dataset,
    steps_per_epoch=20,
    epochs=400,
    verbose=1,
    validation_data=test_dataset,
    validation_freq=1,
    validation_steps=5,
    callbacks=[checkpoint, tb_callback]
)
