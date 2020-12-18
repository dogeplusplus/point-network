import glob
import numpy as np
import tensorflow as tf
import pandas as pd
from plusplus.models.sem_seg_model import SEM_SEG_Model
from visualiser import read_point_cloud

pc = read_point_cloud('/home/albert/github/point-network/data/SHREC/train/5D4KX7RD.txt')

def read_point_cloud(path):
    point_cloud = pd.read_csv(path, sep=' ')
    point_cloud.rename(columns={'//X': 'X'}, inplace=True)
    return point_cloud[['X', 'Y', 'Z']].to_numpy(), point_cloud['label'].to_numpy()


train_path = '/home/albert/github/point-network/data/SHREC/train/*.txt'
train_dataset = dataset_generator(train_path)
val_dataset = dataset_generator('/home/albert/github/point-network/data/SHREC/test/*.txt')

model = SEM_SEG_Model(2, 5)
model.fit(train_dataset)


def dataset_generator(pc_paths, batch_size=2, buffer_size=100):
    dataset = tf.data.Dataset.list_files(pc_paths)
    dataset = dataset.map(lambda file: tf.py_function(read_point_cloud, inp=[file], Tout=[tf.string]))
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y)))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)

    return dataset
