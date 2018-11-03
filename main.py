from utils import read_data
from config import *

def main():
    X_train, Y_train = read_data(TRAIN_PATH, NUM_POINTS)
    X_test, Y_test = read_data(TEST_PATH, NUM_POINTS)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

if __name__ == "__main__":
    main()