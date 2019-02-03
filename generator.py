import keras
import numpy as np
import projections
from utils import read_raw_data


class ViewGenerator(keras.utils.Sequence):

    def __init__(self, raw_path):
        self.shuffle = True
        self.pointcloud = read_raw_data(raw_path, 2048)
        self.labels = self.pointcloud[1]
        self.pointcloud = self.pointcloud[0]
        self.batch_size = 128
        self.channels = 6
        self.indexes = np.arange(0, self.pointcloud.shape[0])
        self.dim = 64
        self.multiplier = 1
        self.n_classes = 40

    def __len__(self):
        return int(np.floor((len(self.pointcloud)*self.multiplier) / self.batch_size))

    def __data_generation(self, indexes):

        subset = self.pointcloud[indexes]
        subset_lables = self.labels[indexes]

        X = np.empty((self.batch_size, self.channels, self.dim, self.dim))
        y = np.empty((self.batch_size, 1), dtype = int)

        for i in range(0, len(subset)):
            X[i,:,:,:] = projections.rasterize(subset[i], self.dim, self.dim, projections.planes)
            y[i, :] = subset_lables[i]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getitem__(self, index):
        # index to the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y