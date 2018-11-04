import sklearn.metrics
import keras.utils
from utils import read_data, plot_history
from config import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

# def main():
X_train, Y_train = read_data(TRAIN_PATH, NUM_POINTS)
X_test, Y_test = read_data(TEST_PATH, NUM_POINTS)
labels = np_utils.to_categorical(Y_train, num_classes=40)

print(X_train.shape)
print(Y_train.shape)

model = Sequential()
model.add(Dense(512, input_dim=64 * 64 * 6))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('softmax'))

model.summary()

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

early_stopping = EarlyStopping(monitor="val_loss", patience=10)
model_checkpoint = ModelCheckpoint("saved_models/best_model.h5", save_best_only=True, verbose=1)
history = model.fit(X_train.reshape(-1, 64 * 64 * 6), labels, epochs=50, batch_size=2048,
    validation_split=0.1, callbacks=[early_stopping, model_checkpoint])

test_labels = np_utils.to_categorical(Y_test, num_classes=40)
test_pred = model.predict(X_test.reshape(-1, 64 * 64 *6))
test_pred = np_utils.to_categorical(test_pred.argmax(axis=-1), 40)

print("test accuracy: ",sklearn.metrics.accuracy_score(test_labels, test_pred))
plot_history(history)



# if __name__ == "__main__":
#     main()