import tensorflow as tf
import numpy as np


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def point2example(point_cloud: np.array, label: int):
    feature = {
        'points': _float_list_feature(point_cloud.flatten()),
        'label': _int64_feature(label)
    }

    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()


def example2point(example_proto: tf.train.Feature):
    feature_desc = {
        'points': tf.io.VarLenFeature([], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_desc)
    parsed_features['points'] = tf.reshape(parsed_features['points'], [None, 3])
    # Shuffle to make ensure model does not over-fit to the permutation
    parsed_features['points'] = tf.random.shuffle(parsed_features['points'])
    return parsed_features


# TODO: train a pointnet++ network using these tfrecords
# TODO: compile C++ files for pointnet++, need to replace CUDA_ROOT with the right path or something
def tfrecords_serializer(point_clouds, labels, dest, clouds_per_record):
    i = 0
    writer = tf.io.TFRecordWriter(path=f'{dest}_{i:02d}.tfrecords')
    for point_cloud, label in zip(point_clouds, labels):
        example = point2example(point_cloud, label)
        writer.write(example)
        i += 1
        if i % clouds_per_record == 0:
            writer = tf.io.TFRecordWriter(path=f'{dest}_{i//clouds_per_record:02d}.tfrecords')
