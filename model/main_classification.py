# Main function for classification model

import sys
from main_utils import argument_parser
from main_utils import load_training_set, load_dev_test_set
from main_utils import show5Results, show1Result, showBestResult, showBest5Result, show5Comparison
import os
from classification_model_L2 import classification_8layers_model
from train_evaluate import train_evaluate
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
    save_dir = "/Users/apple/Desktop/CS230/Project/Report/poster/result/"
    index = np.array([85, 86, 89, 75, 79])
    X = dev_L[index]
    Y = dev_bins[index]
    show5Comparison(train_evaluate, X, Y, dev_L[index], dev_bins[index], dev_ab[index], last_path, annealed = True, annealed_T = 0.89)
    
    plt.figure()
    index = np.array([74, 70, 65, 69, 63])
    X = dev_L[index]
    Y = dev_bins[index]
    show5Comparison(train_evaluate, X, Y, dev_L[index], dev_bins[index], dev_ab[index], last_path, annealed = True, annealed_T = 0.89, save_dir = None)
    plt.show()
    
    plt.figure()
    index = np.array([36, 20, 17, 5, 4])
    
    X = dev_L[index]
    Y = dev_bins[index]
    show5Comparison(train_evaluate, X, Y, dev_L[index], dev_bins[index], dev_ab[index], last_path, annealed = True, annealed_T = 0.89, save_dir = save_dir)
    
    plt.show()