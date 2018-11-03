
from utils import read_data
from config import *
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import sklearn.metrics

# def main():
X_train, Y_train = read_data(TRAIN_PATH, NUM_POINTS)
X_test, Y_test = read_data(TEST_PATH, NUM_POINTS)
labels = np_utils.to_categorical(Y_train, num_classes=40)

print(X_train.shape)
print(Y_train.shape)

model = Sequential()
model.add(Dense(128, input_dim=64 * 64 * 6))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.fit(X_train.reshape(-1, 64 * 64 * 6), labels, epochs=100, batch_size=32)

test_labels = np_utils.to_categorical(Y_test, num_classes=40)
test_pred = model.predict(X_test.reshape(-1, 64 * 64 *6))
test_pred = np_utils.to_categorical(test_pred.argmax(axis=-1), 40)

print("test accuracy: ",sklearn.metrics.accuracy_score(test_labels, test_pred))


# if __name__ == "__main__":
#     main()