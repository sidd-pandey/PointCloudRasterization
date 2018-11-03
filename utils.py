import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from keras.utils import np_utils

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label

def read_data(path, num_points, num_classes):
    points = None
    labels = None
    filenames = [filename for filename in os.listdir(path) if filename.endswith('.h5')]
    for filename in filenames:
        cur_points, cur_labels = load_h5(path + filename)
        cur_points = cur_points.reshape(1, -1, 3)
        cur_labels = cur_labels.reshape(1, -1)
        if labels is None or points is None:
            labels = cur_labels
            points = cur_points
        else:
            labels = np.hstack((labels, cur_labels))
            points = np.hstack((points, cur_points))
    X = points.reshape(-1, num_points, 3)
    Y = labels.reshape(-1, 1)
    Y = np_utils.to_categorical(Y, num_classes)
    return X, Y
    