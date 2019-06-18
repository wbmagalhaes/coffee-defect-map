import tensorflow as tf
import numpy as np
import sys
import os
import cv2 as cv

# int64 is used for numeric values


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# float is used for numeric values


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# bytes is used for string/char values


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def write_tfrecords(filepath, imgs_data):
    # Initiating the writer and creating the tfrecords file.
    writer = tf.python_io.TFRecordWriter(filepath)

    for img_data in imgs_data:
        img = img_data['img']
        dmap = img_data['map']
        h, w = img.shape

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': int64_feature(h),
            'width': int64_feature(w),
            'img': bytes_feature(tf.compat.as_bytes(img.tostring())),
            'map': bytes_feature(tf.compat.as_bytes(dmap.tostring())),
        }))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def get_dataset(filenames, batch_size=10000, shuffle=True):
    print('Config dataset.')
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(serialized_example):
        features = tf.parse_single_example(serialized_example, features={
            'height': tf.FixedLenFeature(shape=(1), dtype=tf.int64),
            'width': tf.FixedLenFeature(shape=(1), dtype=tf.int64),
            'img': tf.FixedLenFeature(shape=(1), dtype=tf.string),
            'map': tf.FixedLenFeature(shape=(1), dtype=tf.string),
        })

        height = tf.cast(features['height'], tf.int64)[0]
        width = tf.cast(features['width'], tf.int64)[0]
        image = tf.decode_raw(features['img'], tf.float32)
        dmap = tf.decode_raw(features['map'], tf.float32)

        image = tf.reshape(image, (height, width, 1), name="image")
        dmap = tf.reshape(dmap, (height, width, 1), name="dmap")

        return image, dmap

    dataset = dataset.map(parser)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size)

    return dataset


def get_data(filenames, shuffle=True):
    dataset = get_dataset(filenames, shuffle=True)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    print('Reading dataset.')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)

        images, dmaps = sess.run(next_element)

    print('End of dataset.')
    return images, dmaps
