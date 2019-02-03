import sklearn.metrics
import keras.utils
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
from keras.layers.core import Lambda
from utils import read_data, save_model_history
from keras.utils import np_utils
from config import *
import os

FOLDER_NAME = "mvcnn"
if not os.path.exists("saved_models/" + FOLDER_NAME):
    os.makedirs("saved_models/" + FOLDER_NAME)

def view_pool(x):
    return K.max(x, axis=1, keepdims=True)

def view_pool_shape(input_shape):
    shape = list(input_shape)
    shape[1] = 1
    return tuple(shape)

keras.backend.set_image_data_format('channels_first')
X_train, Y_train = read_data(TRAIN_PATH, NUM_POINTS)
X_test, Y_test = read_data(TEST_PATH, NUM_POINTS)
labels = np_utils.to_categorical(Y_train, num_classes=NUM_CLASSES)

print(X_train.shape)
print(Y_train.shape)

model = Sequential()

model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(6, IMAGE_WIDTH, IMAGE_HEIGHT)))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Lambda(view_pool, output_shape=view_pool_shape))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(40, activation='softmax'))

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor="val_loss", patience=10)
model_checkpoint = ModelCheckpoint("saved_models/" + FOLDER_NAME + "/best_model.h5", save_best_only=True, verbose=1)
history = model.fit(X_train, labels, epochs=50, batch_size=256,
    validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

test_labels = np_utils.to_categorical(Y_test, num_classes=NUM_CLASSES)

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=-1)
y_pred = np_utils.to_categorical(y_pred, num_classes=NUM_CLASSES)

score = model.evaluate(x=X_test, y=test_labels)
print("test loss & accuracy: ", score)

save_model_history(FOLDER_NAME, history, model, test_labels, y_pred)