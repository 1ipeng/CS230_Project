import sys
from utils import Params, plotLabImage
from main_utils import argument_parser, load_training_set, load_dev_test_set
import matplotlib.pyplot as plt
import os
from classification_model_L2 import classification_8layers_model
from train_evaluate import train_evaluate
from transfer_learning_model import transfer_learning_model
import numpy as np

args = argument_parser(sys.argv)

# Load data
# 50,000/5,000/5,000
params = Params("../experiments/base_model/params.json")
train_L, train_ab, train_bins, train_grayRGB = load_training_set(args, seed = 314)
dev_L, dev_ab, dev_bins, dev_grayRGB, test_L, test_ab, test_bins, test_grayRGB = load_dev_test_set(args, seed = 314)

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

# Show result
def showBestResult(X, dev_L, dev_bins, dev_ab, save_path):
    plt.figure()
    predict_bins, predict_ab, predict_costs, predict_logits = train_evaluate.predict(X, dev_bins, dev_ab, save_path)
    index_min = np.argmin(predict_costs)
    plotLabImage(dev_L[index_min], dev_ab[index_min], (2, 1, 1))
    plotLabImage(dev_L[index_min], predict_ab[index_min], (2, 1, 2))
    print(predict_costs[index_min])
    plt.show()

def show5Results(X, dev_L, dev_bins, dev_ab, start_index, save_path):
    plt.figure()
    predict_bins, predict_ab, predict_costs, predict_logits = train_evaluate.predict(X[start_index:start_index + 5], dev_bins[start_index:start_index + 5], dev_ab[start_index:start_index + 5], save_path)
    count = 0
    for i in range(5):
        count = count + 1
        orig_img = plotLabImage(dev_L[start_index + i], dev_ab[start_index + i], (5, 2, count))
        count = count + 1
        predict_img = plotLabImage(dev_L[start_index + i], predict_ab[i], (5, 2, count))
    print(predict_costs)
    print(predict_logits)
    plt.show()

def show1Result(X, dev_L, dev_bins, dev_ab, start_index, save_path):
    plt.figure()
    predict_bins, predict_ab, predict_cost, predict_logits = train_evaluate.predict(X[start_index:start_index + 1], dev_bins[start_index:start_index + 1], dev_ab[start_index:start_index + 1], save_path)
    orig_img = plotLabImage(dev_L[start_index], dev_ab[start_index], (2, 2, 1))
    predict_img = plotLabImage(dev_L[start_index], predict_ab[0], (2, 2, 2))
    print(predict_bins[:,0,:,:])
    print(dev_bins[:,0,:,:])
    print(predict_logits[:,0,0,:])
    print(predict_cost)
    plt.show()

if args.predict:
	# showBestResult(dev_grayRGB, dev_L, dev_bins, dev_ab)
	# showResult(dev_grayRGB, dev_L, dev_bins, dev_ab, 20, last_path)
	# show5Results(dev_grayRGB, dev_L, dev_bins, dev_ab, 10, best_path)
	show5Results(train_grayRGB, train_L, train_bins, train_ab, 10, last_path)
	# show1Result(dev_grayRGB, dev_L, dev_bins, dev_ab, 10, last_path)
	# show1Result(train_grayRGB, train_L, train_bins, train_ab, 10, last_path)

