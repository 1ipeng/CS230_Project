import sys
from utils import Params, plotLabImage
from main_utils import argument_parser, load_training_set, load_dev_test_set, show5Results, show1Result, showBestResult
import os
from classification_model_L2 import classification_8layers_model
from train_evaluate import train_evaluate
from transfer_learning_model import transfer_learning_model
import numpy as np

args = argument_parser(sys.argv)

# Load data
# 50,000/5,000/5,000
params = Params("../experiments/base_model/params.json")
train_L, train_ab, train_bins, train_grayRGB = load_training_set(args)
dev_L, dev_ab, dev_bins, dev_grayRGB, test_L, test_ab, test_bins, test_grayRGB = load_dev_test_set(args)

# Weight directory
model_dir = "./weights_transfer_learning"
if not os.path.exists(model_dir):
	os.mkdir(model_dir)
best_path = os.path.join(model_dir, 'best_weights')
last_path = os.path.join(model_dir, 'last_weights')

# Build model
train_evaluate = train_evaluate(params, transfer_learning_model, "vgg16_weights.npz")

# Train and predict
if args.train:
	if args.restore:
	    train_evaluate.train(train_grayRGB, train_bins, dev_grayRGB, dev_bins, model_dir, last_path)
	else:
	    train_evaluate.train(train_grayRGB, train_bins, dev_grayRGB, dev_bins, model_dir)

if args.predict:
    X = dev_grayRGB
    Y = dev_bins
    # showBestResult(train_evaluate, X, Y, dev_L, dev_bins, dev_ab, best_path)
    # show5Results(train_evaluate, X, Y, dev_L, dev_bins, dev_ab, 10, best_path)
    # show1Result(train_evaluate, X, Y, dev_L, dev_bins, dev_ab, 0, best_path)

    X = train_grayRGB
    Y = train_bins
    # showBestResult(train_evaluate, X, Y, train_L, train_bins, train_ab, best_path)
    # show5Results(train_evaluate, X, Y, train_L, train_bins, train_ab, 10, best_path)
    show1Result(train_evaluate, X, Y, train_L, train_bins, train_ab, 0, best_path)
