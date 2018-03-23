# Main function for transfer learning model

import sys
from utils import Params, plotLabImage
from main_utils import argument_parser, load_training_set, load_dev_test_set, show5Results, show1Result, showBestResult, show5Comparison
import os
from classification_model_L2 import classification_8layers_model
from train_evaluate import train_evaluate
from transfer_learning_model import transfer_learning_model
import numpy as np
import matplotlib.pyplot as plt

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
    index = np.array([85, 86, 89, 75, 79])
    X = dev_grayRGB[index]
    Y = dev_bins[index]
    show5Comparison(train_evaluate, X, Y, dev_L[index], dev_bins[index], dev_ab[index], last_path, annealed = True, annealed_T = 0.89)
    
    plt.figure()
    index = np.array([74, 70, 65, 69, 63])
    X = dev_grayRGB[index]
    Y = dev_bins[index]
    show5Comparison(train_evaluate, X, Y, dev_L[index], dev_bins[index], dev_ab[index], last_path, annealed = True, annealed_T = 0.89, save_dir = None)
    
    plt.figure()
    index = np.array([36, 20, 17, 5, 4])
    X = dev_grayRGB[index]
    Y = dev_bins[index]
    show5Comparison(train_evaluate, X, Y, dev_L[index], dev_bins[index], dev_ab[index], last_path, annealed = True, annealed_T = 0.89, save_dir = None)
    
    plt.show()