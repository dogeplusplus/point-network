import argparse
import numpy as np
from data_processing.tfrecords import tfrecords_serializer


def parse_arguments():
    parser = argparse.ArgumentParser('Convert point cloud numpy arrays into TFRecords')
    parser.add_argument('--point-clouds', type=str, help='Path to point cloud numpy array')
    parser.add_argument('--labels', type=str, help='Path to corresponding labels')
    parser.add_argument('--dest', type=str, help='Name of the TFRecord')
    parser.add_argument('--clouds-per-record', type=int, help='Number of point clouds to store in each TFRecord',
                        default=500)
    args = parser.parse_args()
    return args

def main(args):
    point_clouds = np.load(args.point_clouds)
    labels = np.load(args.labels)
    tfrecords_serializer(point_clouds, labels, args.dest, args.clouds_per_record)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
