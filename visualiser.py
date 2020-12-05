import laspy
import numpy as np
import pandas as pd
from vedo import Points, show
from pylab import get_cmap


def display_lidar(path, num_points=None):
    file = laspy.file.File(path, mode='r')
    point_cloud = np.stack([file.X, file.Y, file.Z], axis=-1)
    scale = file.get_header().get_scale()
    offset = file.get_header().get_offset()

    point_cloud = point_cloud * scale + offset
    rgb = np.stack([file.Red, file.Green, file.Blue], axis=-1) / 256
    alpha = np.ones((len(point_cloud), 1)) * 255
    rgba = np.concatenate([rgb, alpha], axis=-1)

    if num_points:
        idx = np.random.choice(list(range(len(point_cloud))), num_points, replace=False)
        point_cloud = point_cloud[idx]
        rgba = rgba[idx]

    point_vedo = Points(point_cloud, c=rgba)
    show(point_vedo, axes=True)


def read_point_cloud(path):
    point_cloud = pd.read_csv(path, sep=' ')
    point_cloud.rename(columns={'//X': 'X'}, inplace=True)
    return point_cloud[['X', 'Y', 'Z']].to_numpy(), point_cloud['label'].to_numpy()


def display_point_cloud(point_cloud, label):
    point_vedo = Points(point_cloud[['X', 'Y', 'Z']])
    labels = label
    cmap = get_cmap('Spectral')
    point_vedo.cmap(cmap, labels)
    show(point_vedo, axes=True)


if __name__ == "__main__":
    import os

    BASE_DIR = 'data/SHREC/raw'
    LABELS_DIR = 'data/SHREC/train'
    files = os.listdir(BASE_DIR)

    idx = 25
    raw_cloud = os.path.join(BASE_DIR, files[idx])
    point_path = os.path.join(LABELS_DIR, files[idx].replace('.laz', '.txt'))
    point_cloud, label = read_point_cloud(point_path)
    display_point_cloud(point_cloud, label)
