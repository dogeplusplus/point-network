import os
import json
import glob
import trimesh
import numpy as np
import tensorflow as tf

DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True
)

DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")


# mesh = trimesh.load(os.path.join(DATA_DIR, "chair/train/chair_0001.off"))
# mesh.show()
#
# points = mesh.sample(2048)
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])
# ax.set_axis_off()
# plt.show()


def parse_dataset(num_points=2048):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        class_map[i] = folder.split(os.path.sep)[-1]
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map
    )


NUM_POINTS = 2048
train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS)

np.save('../train_points.npy', train_points)
np.save('../test_points.npy', test_points)
np.save('../train_labels.npy', train_labels)
np.save('../test_labels.npy', test_labels)

with open('../class_map.json', 'w') as f:
    json.dump(CLASS_MAP, f)
