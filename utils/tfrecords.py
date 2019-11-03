import tensorflow as tf


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write(filename, data):
    writer = tf.data.experimental.TFRecordWriter(filename)

    def serialize_example(image, d_map):
        h, w = image.shape

        image = tf.compat.as_bytes(image.tostring())
        d_map = tf.compat.as_bytes(d_map.tostring())

        feature = {
            'im_h': int64_feature(h),
            'im_w': int64_feature(w),
            'image': bytes_feature(image),
            'd_map': bytes_feature(d_map)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def generator():
        for features in data:
            yield serialize_example(*features)

    dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
    writer.write(dataset)


def read(filenames):
    raw_dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        'im_h': tf.io.FixedLenFeature((1), tf.int64),
        'im_w': tf.io.FixedLenFeature((1), tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'd_map': tf.io.FixedLenFeature([], tf.string)
    }

    def parser(example_proto):
        features = tf.io.parse_single_example(example_proto, feature_description)

        h = tf.cast(features['im_h'], tf.int64)[0]
        w = tf.cast(features['im_w'], tf.int64)[0]

        raw_image = tf.io.decode_raw(features['image'], tf.uint8)
        raw_d_map = tf.io.decode_raw(features['d_map'], tf.float32)

        raw_image = tf.cast(raw_image, tf.float32)

        image = tf.reshape(raw_image, (h, w, 1), name="image")
        d_map = tf.reshape(raw_d_map, (h, w, 1), name="d_map")

        return image, d_map

    dataset = raw_dataset.map(parser, num_parallel_calls=4)
    return dataset
