import numpy as np
from config import *
import sklearn.metrics
from utils import read_data
from keras.models import load_model
from keras.utils import np_utils


X_test, Y_test = read_data(TEST_PATH, NUM_POINTS)

model1 = load_model('saved_models/dense-ensemble/best_model-1.h5')
model2 = load_model('saved_models/dense-ensemble/best_model-2.h5')
model3 = load_model('saved_models/dense-ensemble/best_model-3.h5')
predict1 = model1.predict(np.expand_dims(X_test[:, 0, :, :], axis = 1).reshape(-1, IMAGE_WIDTH * IMAGE_HEIGHT * 1))
predict2 = model2.predict(np.expand_dims(X_test[:, 1, :, :], axis = 1).reshape(-1, IMAGE_WIDTH * IMAGE_HEIGHT * 1))
predict3 = model3.predict(np.expand_dims(X_test[:, 2, :, :], axis = 1).reshape(-1, IMAGE_WIDTH * IMAGE_HEIGHT * 1))

predict = np.stack((predict1, predict2, predict3), axis = 1)
predict = predict.max(axis=1)
predict = predict.argmax(axis=-1)

y_pred = np_utils.to_categorical(predict, num_classes=NUM_CLASSES)
y_true = np_utils.to_categorical(Y_test, num_classes=NUM_CLASSES)

print(sklearn.metrics.accuracy_score(y_true, y_pred))
print("\n")
print(sklearn.metrics.classification_report(y_true, y_pred))