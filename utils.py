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
    
def plane_xequal1(point):
    return (point[1], point[2], 1 - point[0])
def plane_yequal1(point):
    return (point[0], point[2], 1 - point[1])
def plane_zequal1(point):
    return (point[0], point[1], 1 - point[2])
def plane_xequal_neg1(point):
    return (point[1], point[2], abs(-1 - point[0]))
def plane_yequal_neg1(point):
    return (point[0], point[2], abs(-1 - point[1]))
def plane_zequal_neg1(point):
    return (point[0], point[1], abs(-1 - point[2]))

planes = [plane_xequal1, plane_yequal1, plane_zequal1, plane_xequal_neg1, plane_yequal_neg1, plane_zequal_neg1]

def rasterize(point_cloud, img_width, img_height, planes):
    channels = len(planes)
    projections = []
    for plane in planes:
        projection = [plane(point) for point in point_cloud]
        projections.append(projection)
    projections = np.array(projections)
    projections[:,:,0:2] = (projections[:,:,0:2] * (img_width/2) + (img_width/2)).astype(np.int16)
    projections[:,:,2] = projections[:,:,2]/2
    
    img = np.zeros((channels, img_width, img_height))
    for i in range(channels):
        projection = projections[i]
        rev_intensity = projection[projection[:,2].argsort()]
        rev_intensity = rev_intensity[::-1]
        for point in rev_intensity:
            img[i][int(point[0])][int(point[1])] = 1 - point[2]
    return img

# index = 500
# point_cloud = train_points_r[index]
# img = rasterize(point_cloud, 64, 64, planes)