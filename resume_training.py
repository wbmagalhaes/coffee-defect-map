import tensorflow as tf
import numpy as np
import time

from utils import config
from utils.tfrecords import get_data
from utils.data_augment import aug_data

import matplotlib.pyplot as plt

model_id = 'CoffeeUNet18'
checkpoint = 2500

print('Using model', model_id)

training_dir = config.CHECKPOINT_DIR + model_id

with tf.name_scope('dataset_load'):
    train_x, train_y = get_data(filenames=[config.TRAINING_PATH], shuffle=True)
    test_x, test_y = get_data(filenames=[config.TESTING_PATH], shuffle=True)

with tf.Session(graph=tf.Graph()) as sess:
    ckpt = '{}/model-{}.meta'.format(training_dir, checkpoint)
    saver = tf.train.import_meta_graph(ckpt, clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(training_dir))
    print('Model loaded.')

    graph = tf.get_default_graph()
    graph_def = tf.get_default_graph().as_graph_def()

    x = graph.get_tensor_by_name("inputs/img_input:0")
    y = graph.get_tensor_by_name("inputs/map_input:0")

    augument_op = aug_data(x, y)

    dmap = graph.get_tensor_by_name("result/dmap:0")
    count = graph.get_tensor_by_name("result/count:0")

    map_loss_op = graph.get_tensor_by_name("score/map_loss_op:0")
    cnt_loss_op = graph.get_tensor_by_name("score/cnt_loss_op:0")

    global_step = tf.train.get_or_create_global_step()

    step_per_sec = graph.get_tensor_by_name("step_per_sec_op:0")
    train_map_op = graph.get_tensor_by_name("optimizer/train_map:0")

    merged = graph.get_tensor_by_name("merged_op:0")
    saver = tf.train.Saver()

    train_writer = tf.summary.FileWriter(training_dir + '/{}/train'.format(checkpoint), sess.graph)
    test_writer = tf.summary.FileWriter(training_dir + '/{}/test'.format(checkpoint))

    time_i = time.time()

    print('Resuming train...')
    for epoch in range(checkpoint + 1, config.EPOCHS + 1):
        delta_time = time.time() - time_i
        time_i = time.time()

        if delta_time <= 0:
            delta_time = 1
        s_per_sec = 1.0 / delta_time

        p = np.random.permutation(len(train_x))[:config.BATCH_SIZE]
        batch_x = train_x[p]
        batch_y = train_y[p]

        feed_dict = {x: batch_x, y: batch_y}
        aug_x, aug_y = sess.run(augument_op, feed_dict=feed_dict)

        feed_dict = {x: aug_x, y: aug_y, step_per_sec: s_per_sec}

        summary, _ = sess.run([merged, train_map_op], feed_dict=feed_dict)
        train_writer.add_summary(summary, epoch)

        if epoch % 10 == 0:
            p = np.random.permutation(len(test_x))[:config.BATCH_SIZE]
            batch_x = test_x[p]
            batch_y = test_y[p]

            feed_dict = {x: batch_x, y: batch_y}
            aug_x, aug_y = sess.run(augument_op, feed_dict=feed_dict)

            feed_dict = {x: aug_x, y: aug_y, step_per_sec: s_per_sec}
            summary, map_loss, cnt_loss = sess.run([merged, map_loss_op, cnt_loss_op], feed_dict=feed_dict)

            test_writer.add_summary(summary, epoch)
            print('epoch: {} map loss: {:.3f} count error: {:.2f} s/step: {:.3f}'.format(epoch, map_loss, cnt_loss * 100, delta_time))

        if epoch % config.CHECKPOINT_INTERVAL == 0:
            saver.save(sess, training_dir + '/model', global_step=epoch)
            saver.export_meta_graph(training_dir + '/model-{}.meta'.format(epoch))

    saver.save(sess, training_dir + '/model', global_step=config.EPOCHS)
    saver.export_meta_graph(training_dir + '/model-{}.meta'.format(config.EPOCHS))

    train_writer.close()
    test_writer.close()

print('Training Finished.')
