import numpy as np
import sklearn.metrics
import keras.utils
import keras
import h5py
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.layers.core import Lambda
from keras.utils import np_utils
import utils
import os
keras.backend.set_image_data_format('channels_first')


TRAIN_PATH = "gdrive/My Drive/colab_notebooks/PointCloud/data/train/train_2048_64.h5"
TEST_PATH = "gdrive/My Drive/colab_notebooks/PointCloud/data/test/test_2048_64.h5"
CHECKPOINT_PATH = "gdrive/My Drive/colab_notebooks/data/point_cloud/"
NUM_POINTS = 2048
NUM_CLASSES = 40

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

X_train, Y_train = utils.read_data(TRAIN_PATH, NUM_POINTS)
X_test, Y_test = utils.read_data(TEST_PATH, NUM_POINTS)
Y_train_ohe = np_utils.to_categorical(Y_train, num_classes=NUM_CLASSES)
Y_test_ohe = np_utils.to_categorical(Y_test, num_classes=NUM_CLASSES)

inputs_layers = []
for i in range(X_train.shape[1]):
    inputs_layers.append(Input(shape=(1, IMAGE_WIDTH, IMAGE_HEIGHT)))

X_train_arr = []
X_test_arr = []
for i in range(X_train.shape[1]):
    X_train_arr.append(np.expand_dims(X_train[:,i,:,:], axis=1))
    X_test_arr.append(np.expand_dims(X_test[:,i,:,:], axis=1))


conv_1 = Conv2D(32, (2, 2), activation='relu')
conv_2 = Conv2D(32, (2, 2), activation='relu')
maxpool_1 = MaxPooling2D(pool_size=(2, 2))
dropout_1 = Dropout(0.25)
conv_3 = Conv2D(32, (2, 2), activation='relu')
conv_4 = Conv2D(32, (2, 2), activation='relu')
maxpool_2 = MaxPooling2D(pool_size=(2, 2))
dropout_2 = Dropout(0.25)

def get_output(input_i):
    X = conv_1(input_i)
    X = conv_2(X)
    X = maxpool_1(X)
    X = dropout_1(X)
    X = conv_3(X)
    X = conv_4(X)
    X = maxpool_2(X)
    X = dropout_2(X)
    return X

intermediate_ouptus = []
for i in range(X_train.shape[1]):
    intermediate_ouptus.append(get_output(inputs_layers[i]))
    
concatenate_layer = keras.layers.merge.concatenate(intermediate_ouptus, axis=1)

conv_5 = Conv2D(192, kernel_size=(1,1), strides=(1, 1), activation="relu")(concatenate_layer)
conv_6 = Conv2D(128, kernel_size=(1,1), strides=(1, 1), activation="relu")(conv_5)

flatten_1 = Flatten()(conv_6)
dense_1 = Dense(512, activation='relu')(flatten_1)
dropout_3 = Dropout(0.2)(dense_1)
dense_2 = Dense(256, activation='relu')(dropout_3)
dropout_4 = Dropout(0.2)(dense_2)
dense_3 = Dense(40, activation='softmax')(dropout_4)

model = Model(inputs=inputs_layers, outputs=dense_3)
model.compile(optimizer='adam',
            loss='categorical_crossentropy', metrics=["accuracy"])

model.summary()


model_checkpoint = ModelCheckpoint(CHECKPOINT_PATH+"best_model.h5", save_best_only=True, verbose=1, monitor="val_acc")
history = model.fit(X_train_arr, Y_train_ohe, epochs=50, batch_size=256, validation_split=0.2, callbacks=[model_checkpoint])
y_pred = model.predict(X_test_arr)
y_pred = y_pred.argmax(axis=-1)
y_pred = np_utils.to_categorical(y_pred, num_classes=NUM_CLASSES)


score = model.evaluate(x=X_test_arr, y=Y_test_ohe)
print("test loss & accuracy: ", score)