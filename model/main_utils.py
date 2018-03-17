import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import Params, plotLabImage

def argument_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", help="restore training from last epoch",
                        action="store_true")
    parser.add_argument("--train", help="train model",
                        action="store_true")
    parser.add_argument("--predict", help="show predict results",
                        action="store_true")
    parser.add_argument("--small", help="train on small dataset",
                        action="store_true")
    parser.add_argument("--toy", help="train on toy dataset",
                        action="store_true")
    parser.add_argument("--superlarge", help="train on superlarge dataset",
                        action="store_true")
    if len(argv) < 2:
        parser.print_usage()
        exit()
    else:
        return parser.parse_args()

def load_training_set(args, size = None, seed = None):
    DIR_TRAIN = "../data/lab_result/train_lab/"
    if args.toy:
        DIR_TRAIN = "../data/lab_result/100_train_lab/"  
    if args.superlarge:
        DIR_TRAIN = "../data/lab_result/super_train_lab/"


    if seed is not None:
        np.random.seed(seed)

    if size is None:
        if args.toy:
            size = 100
        elif args.small:
            size = 5000
        elif args.superlarge:
            size = 200000
        else:
            size = 50000


    train_L = np.load(DIR_TRAIN + "L.npy")
    train_ab = np.load(DIR_TRAIN + "ab.npy")
    train_bins = np.load(DIR_TRAIN + "bins.npy")
    train_grayRGB = np.load(DIR_TRAIN + "grayRGB.npy")

    m = train_L.shape[0]
    permutation = list(np.random.permutation(m))
    train_L = train_L[permutation[0:size]]
    train_ab = train_ab[permutation[0:size]]
    train_bins = train_bins[permutation[0:size]]
    train_grayRGB = train_grayRGB[permutation[0:size]]

    return train_L, train_ab, train_bins, train_grayRGB 

def load_dev_test_set(args, dev_size = None, seed = None):
    DIR_TEST = "../data/lab_result/test_lab/"
    if args.toy:
        DIR_TEST = "../data/lab_result/100_test_lab/"
    if args.superlarge:
        DIR_TEST = "../data/lab_result/super_test_lab/"

    if seed is not None:
        np.random.seed(seed)

    if dev_size is None:
        if args.toy:
            dev_size = 30
        elif args.small:
            dev_size = 500
        else:
            dev_size = 5000

    test_dev_L = np.load(DIR_TEST + "L.npy")
    test_dev_ab = np.load(DIR_TEST + "ab.npy")
    test_dev_bins = np.load(DIR_TEST + "bins.npy")
    test_dev_grayRGB = np.load(DIR_TEST + "grayRGB.npy")

    m = test_dev_L.shape[0]
    permutation = list(np.random.permutation(m))
    dev_index = permutation[0:dev_size]
    test_index = permutation[dev_size:]

    # Build dev/test sets
    dev_L = test_dev_L[dev_index]
    dev_ab = test_dev_ab[dev_index]
    dev_bins = test_dev_bins[dev_index]
    dev_grayRGB = test_dev_grayRGB[dev_index]

    test_L = test_dev_L[test_index]
    test_ab = test_dev_ab[test_index]
    test_bins = test_dev_bins[test_index]
    test_grayRGB = test_dev_grayRGB[test_index]
    return dev_L, dev_ab, dev_bins, dev_grayRGB, test_L, test_ab, test_bins, test_grayRGB  


def load_training_dev_test_set(args, dev_size = 2500, seed = None):
    DIR_TRAIN = "../data/lab_result/train_lab/"
    DIR_TEST = "../data/lab_result/test_lab/"

    train_L = np.load(DIR_TRAIN + "L.npy")
    train_ab = np.load(DIR_TRAIN + "ab.npy")
    train_bins = np.load(DIR_TRAIN + "bins.npy")
    train_grayRGB = np.load(DIR_TRAIN + "grayRGB.npy")

    test_dev_L = np.load(DIR_TEST + "L.npy")
    test_dev_ab = np.load(DIR_TEST + "ab.npy")
    test_dev_bins = np.load(DIR_TEST + "bins.npy")
    test_dev_grayRGB = np.load(DIR_TEST + "grayRGB.npy")

    m = test_dev_L.shape[0]
    permutation = list(np.random.permutation(m))
    dev_index = permutation[0:dev_size]
    test_index = permutation[dev_size:2 * dev_size]
    train_index = permutation[2 * dev_size:]

    dev_L = test_dev_L[dev_index]
    dev_ab = test_dev_ab[dev_index]
    dev_bins = test_dev_bins[dev_index]
    dev_grayRGB = test_dev_grayRGB[dev_index]

    test_L = test_dev_L[test_index]
    test_ab = test_dev_ab[test_index]
    test_bins = test_dev_bins[test_index]
    test_grayRGB = test_dev_grayRGB[test_index]

    train_L = np.concatenate((train_L, test_dev_L[train_index]), axis=0)
    train_ab = np.concatenate((train_ab, test_dev_ab[train_index]), axis=0)
    train_bins = np.concatenate((train_bins, test_dev_bins[train_index]), axis=0)
    train_grayRGB = np.concatenate((train_grayRGB, test_dev_grayRGB[train_index]), axis=0)

    return train_L, train_ab, train_bins, train_grayRGB, dev_L, dev_ab, dev_bins, dev_grayRGB, test_L, test_ab, test_bins, test_grayRGB  


# Show result
def showBestResult(train_evaluate, X, Y, dev_L, dev_bins, dev_ab, save_path, annealed = False, annealed_T = 0.32):
    predict_ab, predict_costs, predict_logits, predict_accuracy = train_evaluate.predict(X, Y, save_path, annealed, annealed_T)
    index_min = np.argmin(predict_costs)
    plotLabImage(dev_L[index_min], dev_ab[index_min], (2, 1, 1))
    plotLabImage(dev_L[index_min], predict_ab[index_min], (2, 1, 2))
    print(predict_costs[index_min])
    # plt.show()

def show5Results(train_evaluate, X, Y, dev_L, dev_bins, dev_ab, start_index, save_path, annealed = False, annealed_T = 0.32):
    predict_ab, predict_costs, predict_logits, predict_accuracy = train_evaluate.predict(X[start_index:start_index + 5], Y[start_index:start_index + 5], save_path, annealed, annealed_T)
    count = 0
    for i in range(5):
        count = count + 1
        orig_img = plotLabImage(dev_L[start_index + i], dev_ab[start_index + i], (5, 2, count))
        count = count + 1
        predict_img = plotLabImage(dev_L[start_index + i], predict_ab[i], (5, 2, count))
    print(predict_costs)
    # plt.show()

def show1Result(train_evaluate, X, Y, dev_L, dev_bins, dev_ab, start_index, save_path, annealed = False, annealed_T = 0.32):
    predict_ab, predict_cost, predict_logits, predict_accuracy = train_evaluate.predict(X[start_index:start_index + 1], Y[start_index:start_index + 1], save_path, annealed, annealed_T)
    orig_img = plotLabImage(dev_L[start_index], dev_ab[start_index], (1, 3, 1))
    gray_img = plotLabImage(dev_L[start_index], dev_ab[start_index], (1, 3, 2), grayScale = True)
    predict_img = plotLabImage(dev_L[start_index], predict_ab[0], (1, 3, 3))

    # print(predict_bins[:,0,:,:])
    # print(dev_bins[:,0,:,:])
    # print(predict_logits[:,0,0,:])
    print("cost:", predict_cost)
    print("accuracy:", predict_accuracy)
    # plt.show()


