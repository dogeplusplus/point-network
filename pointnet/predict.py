import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model

NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_POINTS = 2048

train_points = np.load("../data/train_points.npy").astype(np.float32)
train_labels = np.load("../data/train_labels.npy")
test_points = np.load("../data/test_points.npy").astype(np.float32)
test_labels = np.load("../data/test_labels.npy")

with open('../class_map.json', 'r') as f:
    CLASS_MAP = json.load(f)


def augment(points, label):
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float32)
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)


def visualize_predictions(model: Model, dataset: tf.data.Dataset):
    data = dataset.take(1)
    points, labels = list(data)[0]
    points = points[:8]
    labels = labels[:8]

    preds = model(points)
    preds = tf.argmax(preds, axis=-1)
    points = points.numpy()

    fig = plt.figure(figsize=(15, 10))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        ax.set_title(f"pred: {CLASS_MAP[str(preds[i].numpy())]}, label: {CLASS_MAP[str(labels.numpy()[i])]}")
        ax.set_axis_off()
    plt.show()


model = tf.saved_model.load('../model_instances/point_net')
visualize_predictions(model, test_dataset)
