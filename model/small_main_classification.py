import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from utils import Params, bins2ab, plotLabImage, random_mini_batches
import os
from scipy.misc import imsave
from classification_model import classification_8layers, model
# Experiment on a toy dataset with train size 100, dev size 30

parser = argparse.ArgumentParser()
parser.add_argument("--restore", help="restore training from last epoch",
                    action="store_true")
parser.add_argument("--train", help="train model",
                    action="store_true")
parser.add_argument("--predict", help="show predict results",
                    action="store_true")

args = parser.parse_args()
if len(sys.argv) < 2:
    parser.print_usage()

# Load data
DIR_TRAIN = "../data/lab_result/100_train_lab/"
data_L = np.load(DIR_TRAIN + "L.npy")
data_ab = np.load(DIR_TRAIN + "ab.npy")
ab_bins = np.load(DIR_TRAIN + "bins.npy")
params = Params("../experiments/base_model/params.json")

# Shuffle data
train_size = 100
dev_size = 30
m = data_L.shape[0]
np.random.seed(10)
permutation = list(np.random.permutation(m))
train_index = permutation[0:train_size]
dev_index = permutation[train_size:train_size + dev_size]

# Build toy dataset
train_L = data_L[train_index]
train_ab = data_ab[train_index]
train_bins = ab_bins[train_index]
dev_L = data_L[dev_index]
dev_ab = data_ab[dev_index]
dev_bins = ab_bins[dev_index]

model_dir = "./weights_classification"
save_path = os.path.join(model_dir, 'last_weights')

# Build model
model = model(params, classification_8layers)


# Train and predict
if args.train:
	if args.restore:
	    model.train(train_L, train_bins, dev_L, dev_bins, model_dir, save_path)
	else:
	    model.train(train_L, train_bins, dev_L, dev_bins, model_dir)

# Show result
if args.predict:
	save_path = os.path.join(model_dir, 'last_weights')
	predict_bins, predict_ab, predict_cost = model.predict(dev_L[0:5], dev_bins[0:5], dev_ab[0:5], params, save_path)
	count = 0
	for i in range(5):
	    count = count + 1
	    orig_img = plotLabImage(dev_L[i], dev_ab[i], (5, 2, count))
	    count = count + 1
	    predict_img = plotLabImage(dev_L[i], predict_ab[i], (5, 2, count))

	plt.figure()
	predict_bins, predict_ab, predict_cost = model.predict(train_L[0:5], train_bins[0:5], train_ab[0:5], params, save_path)
	count = 0
	for i in range(5):
	    count = count + 1
	    orig_img = plotLabImage(train_L[i], train_ab[i], (5, 2, count))
	    count = count + 1
	    predict_img = plotLabImage(train_L[i], predict_ab[i], (5, 2, count))
	plt.show()
