import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from utils import Params, bins2ab, plotLabImage, random_mini_batches
import os
from scipy.misc import imsave
from classification_model_L2 import classification_8layers_model
from train_evaluate import train_evaluate
from transfer_learning_model import transfer_learning_model


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

args = parser.parse_args()
if len(sys.argv) < 2:
    parser.print_usage()

# Load data
# 50,000/5,000/5,000
DIR_TRAIN = "../data/lab_result/train_lab/"
DIR_TEST = "../data/lab_result/test_lab/"
if args.toy:
	DIR_TRAIN = "../data/lab_result/100_train_lab/"
	DIR_TEST = "../data/lab_result/100_test_lab/"

train_L = np.load(DIR_TRAIN + "L.npy")
train_ab = np.load(DIR_TRAIN + "ab.npy")
train_bins = np.load(DIR_TRAIN + "bins.npy")
train_grayRGB = np.load(DIR_TRAIN + "grayRGB.npy")
params = Params("../experiments/base_model/params.json")

test_dev_L = np.load(DIR_TEST + "L.npy")
test_dev_ab = np.load(DIR_TEST + "ab.npy")
test_dev_bins = np.load(DIR_TEST + "bins.npy")
test_dev_grayRGB = np.load(DIR_TEST + "grayRGB.npy")

# Shuffle data
dev_size = 5000
if args.small:
	dev_size = 500
if args.toy:
	dev_size = 30

m = test_dev_L.shape[0]
np.random.seed(313)
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

if args.small:
	train_L = train_L[0:5000]
	train_ab = train_ab[0:5000]
	train_bins = train_bins[0:5000]
	train_grayRGB = train_grayRGB[0:5000]

	dev_L = dev_L[0:500]
	dev_ab = dev_ab[0:500]
	dev_bins = dev_bins[0:500]
	dev_grayRGB = dev_grayRGB[0:500]

if args.toy:
	train_L = train_L[0:10]
	train_ab = train_ab[0:10]
	train_bins = train_bins[0:10]
	train_grayRGB = train_grayRGB[0:10]

	dev_L = dev_L[1:2]
	dev_ab = dev_ab[1:2]
	dev_bins = dev_bins[1:2]
	dev_grayRGB = dev_grayRGB[1:2]


# Weight directory
model_dir = "./weights_transfer_learning"
if not os.path.exists(model_dir):
	os.mkdir(model_dir)

save_path = os.path.join(model_dir, 'last_weights')
best_path = os.path.join(model_dir, 'best_weights')
last_path = os.path.join(model_dir, 'last_weights')

# Build model
train_evaluate = train_evaluate(params, transfer_learning_model, "vgg16_weights.npz")

# Train and predict
if args.train:
	if args.restore:
	    train_evaluate.train(train_grayRGB, train_bins, dev_grayRGB, dev_bins, model_dir, save_path)
	else:
	    train_evaluate.train(train_grayRGB, train_bins, dev_grayRGB, dev_bins, model_dir)

# Show result
def showBestResult(dev_grayRGB, dev_L, dev_bins, dev_ab, save_path):
	plt.figure()
	predict_bins, predict_ab, predict_costs = train_evaluate.predict(dev_grayRGB, dev_bins, dev_ab, save_path)
	index_min = np.argmin(predict_costs)
	plotLabImage(dev_L[index_min], dev_ab[index_min], (2, 1, 1))
	plotLabImage(dev_L[index_min], predict_ab[index_min], (2, 1, 2))
	print(predict_costs[index_min])
	plt.show()

def show5Results(dev_grayRGB, dev_L, dev_bins, dev_ab, start_index, save_path):
	plt.figure()
	predict_bins, predict_ab, predict_costs = train_evaluate.predict(dev_grayRGB[start_index:start_index + 5], dev_bins[start_index:start_index + 5], dev_ab[start_index:start_index + 5], save_path)
	count = 0
	for i in range(5):
	    count = count + 1
	    orig_img = plotLabImage(dev_L[start_index + i], dev_ab[start_index + i], (5, 2, count))
	    count = count + 1
	    predict_img = plotLabImage(dev_L[start_index + i], predict_ab[i], (5, 2, count))
	print(predict_costs)
	plt.show()

def show1Result(dev_grayRGB, dev_L, dev_bins, dev_ab, start_index, save_path):
	plt.figure()
	predict_bins, predict_ab, predict_cost = train_evaluate.predict(dev_grayRGB[start_index:start_index + 1], dev_bins[start_index:start_index + 1], dev_ab[start_index:start_index + 1], save_path)
	orig_img = plotLabImage(dev_L[start_index], dev_ab[start_index], (2, 2, 1))
	predict_img = plotLabImage(dev_L[start_index], predict_ab[0], (2, 2, 2))
	print(predict_cost)
	plt.show()

if args.predict:
	# showBestResult(dev_grayRGB, dev_L, dev_bins, dev_ab)
	# showResult(dev_grayRGB, dev_L, dev_bins, dev_ab, 20, save_path)
	# show5Results(dev_grayRGB, dev_L, dev_bins, dev_ab, 10, best_path)
	# show5Results(train_grayRGB, train_L, train_bins, train_ab, 10, last_path)
	show1Result(train_grayRGB, train_L, train_bins, train_ab, 0, save_path)
