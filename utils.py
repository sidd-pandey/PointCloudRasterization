import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import sklearn.metrics
import keras.utils
from keras.utils import np_utils
from tqdm import tqdm
from projections import rasterize, planes

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label

def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()

def read_data(path, num_points):
    points, labels = load_h5(path)
    return points, labels

def read_raw_data(path, num_points):
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
    return X, Y

def save_raster_data(read_path, save_path, num_points, img_width, img_height):
    points, labels = read_raw_data(read_path, num_points)
    all_images = []
    for _, point_cloud in enumerate(tqdm(points)):  
        img = rasterize(point_cloud, img_width, img_height, planes)
        all_images.append(img)
    save_h5(save_path, all_images, labels)

def save_train_test_raster_data(read_train_path, save_train_path, read_test_path, save_test_path, 
    num_points, img_width, img_height):
    save_raster_data(read_train_path, save_train_path, num_points, img_width, img_height)
    save_raster_data(read_test_path, save_test_path, num_points, img_width, img_height)

def plot_history(history, path):
    #  "Accuracy"
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path+"/accuracy.png", dpi=300)
    # "Loss"
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path+"/loss.png", dpi=300)

def save_model_history(modelname, history, model, y_true, y_pred):
    path = "saved_models/" + modelname
    if not os.path.exists(path):
        os.makedirs(path)
    # save plots
    plot_history(history, path)
    # save model desc
    keras.utils.plot_model(model, to_file=path+"/model_desc.png", show_shapes=True)
    # save metrics
    summary_file = open(path+"/summary.txt","w", encoding="utf-8")
    summary_file.writelines(sklearn.metrics.classification_report(y_true, y_pred))
    summary_file.write("\n")
    summary_file.write("accuracy_score: " + str(sklearn.metrics.accuracy_score(y_true, y_pred)) + "\n")
    summary_file.close()


if __name__ == "__main__":
    RAW_TRAIN_PATH = "data/train/raw/"
    RAW_TEST_PATH = "data/test/raw/"
    SAVE_TRAIN_PATH = "data/train/"
    SAVE_TEST_PATH = "data/test/"
    NUM_POINTS = 2048
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    save_train = SAVE_TRAIN_PATH + "train_" + str(NUM_POINTS) + "_" + str(IMG_WIDTH) + ".h5"
    save_test = SAVE_TEST_PATH + "test_" + str(NUM_POINTS) + "_" + str(IMG_WIDTH) + ".h5"
    # save_train_test_raster_data(RAW_TRAIN_PATH, save_train, RAW_TEST_PATH, save_test, NUM_POINTS,
    #     IMG_WIDTH, IMG_HEIGHT)
    save_raster_data(RAW_TRAIN_PATH, save_train, NUM_POINTS, IMG_WIDTH, IMG_HEIGHT)

