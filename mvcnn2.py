import numpy as np
import sklearn.metrics
import keras.utils
import keras
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.layers.core import Lambda
from utils import read_data, save_model_history
from keras.utils import np_utils
from config import *
import os

FOLDER_NAME = "mvcnn2"
if not os.path.exists("saved_models/" + FOLDER_NAME):
    os.makedirs("saved_models/" + FOLDER_NAME)

keras.backend.set_image_data_format('channels_first')
X_train, Y_train = read_data(TRAIN_PATH, NUM_POINTS)
X_test, Y_test = read_data(TEST_PATH, NUM_POINTS)
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

print(X_train.shape)
print(Y_train_ohe.shape)

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

flatten_1 = Flatten()(concatenate_layer)
dense_1 = Dense(128, activation='relu')(flatten_1)
dropout_3 = Dropout(0.5)(dense_1)
dense_2 = Dense(40, activation='softmax')(dropout_3)

model = Model(inputs=inputs_layers, outputs=dense_2)
model.compile(optimizer='adam',
            loss='categorical_crossentropy', metrics=["accuracy"])

model.summary()

early_stopping = EarlyStopping(monitor="val_accuracy", patience=10)
model_checkpoint = ModelCheckpoint("saved_models/" + FOLDER_NAME + "/best_model.h5", save_best_only=True, verbose=1)
history = model.fit(X_train_arr, Y_train_ohe, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
y_pred = model.predict(X_test_arr)
y_pred = y_pred.argmax(axis=-1)
y_pred = np_utils.to_categorical(y_pred, num_classes=NUM_CLASSES)

score = model.evaluate(x=X_test_arr, y=Y_test_ohe)
print("test loss & accuracy: ", score)

save_model_history(FOLDER_NAME, history, model, Y_test_ohe, y_pred)


