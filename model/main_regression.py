import sys
from utils import Params, plotLabImage
from main_utils import argument_parser, load_training_set, load_dev_test_set
import matplotlib.pyplot as plt
import os
from classification_model_L2 import classification_8layers_model
from train_evaluate import train_evaluate
from regression_model import regression_8layers_model
import numpy as np

args = argument_parser(sys.argv)

params = Params("../experiments/base_model/params.json")
train_L, train_ab, train_bins, train_grayRGB = load_training_set(args,seed = 110)
dev_L, dev_ab, dev_bins, dev_grayRGB, test_L, test_ab, test_bins, test_grayRGB = load_dev_test_set(args, seed = 110)

# Weight directory
model_dir = "./weights_regression"
if not os.path.exists(model_dir):
	os.mkdir(model_dir)
best_path = os.path.join(model_dir, 'best_weights')
last_path = os.path.join(model_dir, 'last_weights')

# Build model
train_evaluate = train_evaluate(params, regression_8layers_model, model_type = 'regression')

# Train and predict
if args.train:
	if args.restore:
	    train_evaluate.train(train_L, train_ab, dev_L, dev_ab, model_dir, last_path)
	else:
	    train_evaluate.train(train_L, train_ab, dev_L, dev_ab, model_dir)

# Show result
def showBestResult(X, Y, dev_L, dev_bins, dev_ab, save_path):
    plt.figure()
    predict_ab, predict_costs, predict_logits, predict_accuracy, check = train_evaluate.predict(X, Y, save_path)
    index_min = np.argmin(predict_costs)
    plotLabImage(dev_L[index_min], dev_ab[index_min], (2, 1, 1))
    plotLabImage(dev_L[index_min], predict_ab[index_min], (2, 1, 2))
    print(predict_costs[index_min])
    plt.show()

def show5Results(X, Y, dev_L, dev_bins, dev_ab, start_index, save_path):
    plt.figure()
    predict_ab, predict_costs, predict_logits, predict_accuracy, check = train_evaluate.predict(X[start_index:start_index + 5], Y[start_index:start_index + 5], save_path)
    count = 0
    for i in range(5):
        count = count + 1
        orig_img = plotLabImage(dev_L[start_index + i], dev_ab[start_index + i], (5, 2, count))
        count = count + 1
        predict_img = plotLabImage(dev_L[start_index + i], predict_ab[i], (5, 2, count))
    print(predict_costs)
    plt.show()

def show1Result(X, Y, dev_L, dev_bins, dev_ab, start_index, save_path):
    plt.figure()
    predict_ab, predict_cost, predict_logits, predict_accuracy, check = train_evaluate.predict(X[start_index:start_index + 1], Y[start_index:start_index + 1], save_path)
    orig_img = plotLabImage(dev_L[start_index], dev_ab[start_index], (1, 3, 1))
    gray_img = plotLabImage(dev_L[start_index], dev_ab[start_index], (1, 3, 2), grayScale = True)
    predict_img = plotLabImage(dev_L[start_index], predict_ab[0], (1, 3, 3))
    # print(predict_logits[:,0,0,:])
    print("cost:", predict_cost)
    print("accuracy:", predict_accuracy)
    plt.show()

if args.predict:
    # X = dev_L
    # Y = dev_ab
    # showBestResult(X[20:30], Y[20:30], dev_L[20:30], dev_bins[20:30], dev_ab[20:30], last_path)
	# show5Results(dev_L, dev_L, dev_bins, dev_ab, 10, best_path)
    # show1Result(X, Y, dev_L, dev_bins, dev_ab, 20, last_path)

   
    X = train_L
    Y = train_ab
    # showBestResult(X, Y, train_L, train_bins, train_ab, best_path)
    # show5Results(train_L, train_L, train_bins, train_ab, 10, best_path)
    show1Result(X, Y, train_L, train_bins, train_ab, 0, last_path)

    

	