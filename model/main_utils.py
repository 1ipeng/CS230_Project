import argparse
import numpy as np


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
    if len(argv) < 2:
        parser.print_usage()
        exit()
    else:
        return parser.parse_args()

def load_training_set(args, size = None, seed = None):
    DIR_TRAIN = "../data/lab_result/train_lab/"
    if args.toy:
        DIR_TRAIN = "../data/lab_result/100_train_lab/"  

    if seed is not None:
        np.random.seed(seed)

    if size is None:
        if args.toy:
            size = 100
        elif args.small:
            size = 5000
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
    DIR_TEST = "../data/lab_result/100_test_lab/"

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
