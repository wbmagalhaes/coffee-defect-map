import tensorflow as tf
import numpy as np
import time

from utils import config
from utils.tfrecords import get_data
from utils.data_augment import aug_data

from utils.model import map_loss
from utils.model import cnt_loss

import CoffeeUNet18 as cnn

training_dir = config.CHECKPOINT_DIR + cnn.model_id

print('Using model', cnn.model_id)

with tf.name_scope('dataset_load'):
    train_x, train_y = get_data(filenames=[config.TRAINING_PATH], shuffle=True)
    test_x, test_y = get_data(filenames=[config.TESTING_PATH], shuffle=True)

with tf.name_scope('inputs'):
    orig_x = tf.placeholder(
        tf.uint8, [None, None, None, 3])
    orig_y = tf.placeholder(
        tf.float32, [None, None, None, 1])

    augument_op = aug_data(orig_x, orig_y)

    x = tf.placeholder(
        tf.float32, [None, config.IMG_SIZE, config.IMG_SIZE, 3], name='image_input')
    y = tf.placeholder(
        tf.float32, [None, config.IMG_SIZE, config.IMG_SIZE, 1], name='label_input')

with tf.name_scope('neural_net'):
    y_pred = cnn.model(x)

with tf.name_scope('result'):
    dmap = tf.identity(y_pred, name='dmap')
    count = tf.reduce_sum(y_pred, [1, 2], name='count')

with tf.name_scope('score'):
    map_loss_op = map_loss(y_pred, y)
    cnt_loss_op = cnt_loss(count, y)

tf.summary.scalar('score/map_loss', map_loss_op)
tf.summary.scalar('score/cnt_loss', cnt_loss_op)

global_step = tf.train.get_or_create_global_step()

learning_rate = tf.train.exponential_decay(
    learning_rate=config.LEARNING_RATE,
    global_step=global_step,
    decay_steps=3000,
    decay_rate=1, # 0.96,
    staircase=False
)

tf.summary.scalar('learning_rate', learning_rate)

step_per_sec = tf.placeholder(tf.float32)
tf.summary.scalar('step_per_sec', step_per_sec)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.LEARNING_RATE, name='AdamOpt')

    train_map_op = optimizer.minimize(
        map_loss_op, global_step=global_step, name='train_map')
    train_cnt_op = optimizer.minimize(
        cnt_loss_op, global_step=global_step, name='train_cnt')

tf.summary.image('im/_in', x, max_outputs=1)
tf.summary.image('im/_out', y_pred, max_outputs=1)
tf.summary.image('im/expected', y, max_outputs=1)

merged = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(training_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(training_dir + '/test')

    tf.global_variables_initializer().run()

    time_i = time.time()

    print('Starting train...')
    for epoch in range(config.EPOCHS + 1):
        delta_time = time.time() - time_i
        time_i = time.time()

        if delta_time <= 0:
            delta_time = 1
        s_per_sec = 1.0 / delta_time

        p = np.random.permutation(len(train_x))[:config.BATCH_SIZE]
        batch_x = train_x[p]
        batch_y = train_y[p]

        feed_dict = {orig_x: batch_x, orig_y: batch_y}
        aug_x, aug_y = sess.run(augument_op, feed_dict=feed_dict)

        feed_dict = {x: aug_x, y: aug_y, step_per_sec: s_per_sec}
        summary, _ = sess.run([merged, train_map_op], feed_dict=feed_dict)
        train_writer.add_summary(summary, epoch)

        if epoch % 10 == 0:
            p = np.random.permutation(len(test_x))[:config.BATCH_SIZE]
            batch_x = test_x[p]
            batch_y = test_y[p]
            
            feed_dict = {orig_x: batch_x, orig_y: batch_y}
            aug_x, aug_y = sess.run(augument_op, feed_dict=feed_dict)

            feed_dict = {x: aug_x, y: aug_y, step_per_sec: s_per_sec}
            summary, map_loss, cnt_loss = sess.run(
                [merged, map_loss_op, cnt_loss_op], feed_dict=feed_dict)

            test_writer.add_summary(summary, epoch)
            print('epoch: %i map loss: %.3f count error: %.2f s/step: %.3f' % (
                epoch, map_loss, cnt_loss * 100, delta_time))

        if epoch % config.CHECKPOINT_INTERVAL == 0:
            saver.save(sess, training_dir + '/model', global_step=epoch)
            saver.export_meta_graph(
                training_dir + '/model-{}.meta'.format(epoch))

    saver.save(sess, training_dir + '/model', global_step=config.EPOCHS)
    saver.export_meta_graph(
        training_dir + '/model-{}.meta'.format(config.EPOCHS))

    train_writer.close()
    test_writer.close()

print('Training Finished.')
