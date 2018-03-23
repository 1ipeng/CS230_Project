# CS230_Project

## Build Dataset
Download CIFAR-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

Once the download is complte, move the dataset into data/cifar-10-batches-py. Run the script build_dataset.py, which will output the data we need for this project.

To build the whole dataset(50000 training examples, 10000 test examples), run
```
python build_dataset.py big
```
To build the toy dataset(500 training examples, 100 test examples), run
```
python build_dataset.py small
```
For running transfer learning model, the weights file (vgg16_weights.npz) can be downloaded from [here](http://www.cs.toronto.edu/~frossard/post/vgg16/).
Once the download is complte, move the weights into model/vgg16_weights.npz

## Quick Start
### 1. Choose hyperparameters.
Change the hyperparameters in experiments/base_model/params.json.
### 2. Train your experiment. 
To train regression model, run
```
python main_regression.py --train [--toy if use toy dataset]
```
To train classification model, run
```
python main_classification.py --train [--toy if use toy dataset]
```
To train transfer learning model, run
```
python main_transfer_learning.py --train [--toy if use toy dataset]
```
### 3. Display the results.
To show results, run 
```
python main_regression.py --predict [--toy if use toy dataset]
```
```
python main_classification.py --predict [--toy if use toy dataset]
```
```
python main_transfer_learning.py --predict [--toy if use toy dataset]
```