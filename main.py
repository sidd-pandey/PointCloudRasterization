import sklearn.metrics
import keras.utils
from utils import read_data, plot_history, save_model_history, enhance
from config import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

<<<<<<< HEAD
FOLDER_NAME = "dense"
if not os.path.exists("saved_models/" + FOLDER_NAME):
    os.makedirs("saved_models/" + FOLDER_NAME)

=======
>>>>>>> enhancements
X_train, Y_train = read_data(TRAIN_PATH, NUM_POINTS)
X_test, Y_test = read_data(TEST_PATH, NUM_POINTS)
labels = np_utils.to_categorical(Y_train, num_classes=40)

print(X_train.shape)
print(Y_train.shape)

X_train = enhance(X_train)
X_test = enhance(X_test)

model = Sequential()
model.add(Dense(512, input_dim=IMAGE_WIDTH * IMAGE_HEIGHT * 6))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('softmax'))

model.summary()

adam = keras.optimizers.Adam(lr=0.0001)

model.compile(optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

early_stopping = EarlyStopping(monitor="val_loss", patience=10)
model_checkpoint = ModelCheckpoint("saved_models/" + FOLDER_NAME + "/best_model.h5", save_best_only=True, verbose=1)
history = model.fit(X_train.reshape(-1, IMAGE_WIDTH * IMAGE_HEIGHT * 6), labels, epochs=1, batch_size=256,
    validation_split=0.1, callbacks=[early_stopping, model_checkpoint])

test_labels = np_utils.to_categorical(Y_test, num_classes=NUM_CLASSES)

score = model.evaluate(x=X_test.reshape(-1, IMAGE_WIDTH * IMAGE_HEIGHT *6), y=test_labels)

y_pred = model.predict(X_test.reshape(-1, 64 * 64 * 6))
y_pred = y_pred.argmax(axis=-1)
y_pred = np_utils.to_categorical(y_pred, num_classes=NUM_CLASSES)

print("test loss & accuracy: ", score)
save_model_history(FOLDER_NAME, history, model, test_labels, y_pred)