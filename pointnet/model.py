import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dense, Activation, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import GlobalMaxPooling1D, Reshape, Dot
from tensorflow.python.keras.regularizers import Regularizer

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


def conv_bn(x, filters):
    x = Conv1D(filters, kernel_size=1, padding='valid')(x)
    x = BatchNormalization(momentum=0.)(x)
    return Activation('relu')(x)


def dense_bn(x, filters):
    x = Dense(filters)(x)
    x = BatchNormalization(momentum=0.)(x)
    return Activation('relu')(x)


class OrthogonalRegularizer(Regularizer):
    def __init__(self, num_features, l2reg=1e-3):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return {
            'num_features': self.num_features,
            'l2reg': self.l2reg,
        }


def tnet(inputs, num_features):
    bias = Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = Dense(num_features * num_features, kernel_initializer='zeros', bias_initializer=bias, activity_regularizer=reg)(
        x)
    feat_T = Reshape((num_features, num_features))(x)
    return Dot(axes=(2, 1))([inputs, feat_T])


def classification_model():
    inputs = Input(shape=(NUM_POINTS, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = Dropout(0.3)(x)

    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='pointnet')
    return model

def segmentation_model(num_classes):
    inputs = Input(shape=(None, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    concat = tnet(x, 32)
    x = conv_bn(concat, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    global_vector = GlobalMaxPooling1D()(x)
    global_repeat = tf.tile(global_vector[:, tf.newaxis, ...], (1, NUM_POINTS, 1))
    x = tf.keras.layers.Concatenate()([concat, global_repeat])
    x = conv_bn(x, 256)
    x = conv_bn(x, 128)
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    outputs = conv_bn(x, num_classes)

    model = Model(inputs=inputs, outputs=outputs, name='pointnet_seg')
    return model

model = segmentation_model(3)
print(model.summary())
model.save('testy')

# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     metrics=['sparse_categorical_accuracy']
# )
#
# model.fit(train_dataset, epochs=2, validation_data=test_dataset)
# model.save('model_instances/point_net')
