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

    def serialize_example(x, y):
        x = tf.compat.as_bytes(x.tostring())
        y = tf.compat.as_bytes(y.tostring())

        feature = {
            'x': bytes_feature(x),
            'y': bytes_feature(y)
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
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string)
    }

    def parser(example_proto):
        features = tf.io.parse_single_example(example_proto, feature_description)

        raw_x = tf.io.decode_raw(features['x'], tf.uint8)
        raw_y = tf.io.decode_raw(features['y'], tf.float32)

        raw_x = tf.cast(raw_x, tf.float32)

        x = tf.reshape(raw_x, (256, 256, 1), name="x")
        y = tf.reshape(raw_y, (256, 256, 1), name="y")

        return x, y

    dataset = raw_dataset.map(parser, num_parallel_calls=4)
    return dataset
