import sys
from main_utils import argument_parser, load_training_set, load_dev_test_set, show5Results, show1Result, showBestResult
import os
from classification_model_L2 import classification_8layers_model
from train_evaluate import train_evaluate
from transfer_learning_model import transfer_learning_model
import numpy as np
from utils import Params
import matplotlib.pyplot as plt

args = argument_parser(sys.argv)

params = Params("../experiments/base_model/params.json")
train_L, train_ab, train_bins, train_grayRGB = load_training_set(args)
dev_L, dev_ab, dev_bins, dev_grayRGB, test_L, test_ab, test_bins, test_grayRGB = load_dev_test_set(args)

# Weight directory
model_dir = "./weights_classification"
if not os.path.exists(model_dir):
	os.mkdir(model_dir)
best_path = os.path.join(model_dir, 'best_weights')
last_path = os.path.join(model_dir, 'last_weights')

# Build model
train_evaluate = train_evaluate(params, classification_8layers_model)

# Train and predict
if args.train:
	if args.restore:
	    train_evaluate.train(train_L, train_bins, dev_L, dev_bins, model_dir, last_path)
	else:
	    train_evaluate.train(train_L, train_bins, dev_L, dev_bins, model_dir)

if args.predict:
    # X = dev_L
    # Y = dev_bins
    # showBestResult(train_evaluate, X, Y, dev_L, dev_bins, dev_ab, best_path)
    # show5Results(train_evaluate, X, Y, dev_L, dev_bins, dev_ab, 20, last_path)
    # show1Result(train_evaluate, X, Y, dev_L, dev_bins, dev_ab, 20, last_path)

    X = train_L
    Y = train_bins
    # showBestResult(train_evaluate, X, Y, train_L, train_bins, train_ab, best_path)
    show5Results(train_evaluate, X, Y, train_L, train_bins, train_ab, 20, last_path)
    plt.figure()
    show5Results(train_evaluate, X, Y, train_L, train_bins, train_ab, 20, last_path, annealed = True, annealed_T = 0.89)
    # show1Result(train_evaluate, X, Y, train_L, train_bins, train_ab, 99, last_path)
    plt.show()
