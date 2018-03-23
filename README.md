# CS230_Project

## Build Dataset
### 1. Download dataset

Download CIFAR-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

Once the download is complte, move the dataset into data/cifar-10-batches-py. 

### 2. Build dataset for image colorization
To build the whole dataset(50000 training examples, 10000 test examples) for colorization, run
```
python build_dataset.py big
```
To build the toy dataset(500 training examples, 100 test examples) for colorization, run
```
python build_dataset.py small
```
For running transfer learning model, the weights file (vgg16_weights.npz) can be downloaded from [here](http://www.cs.toronto.edu/~frossard/post/vgg16/).
Once the download is complte, move the weights into model/vgg16_weights.npz

### 3. Build dataset for image classification
To build the whole dataset(50000 training examples, 10000 test examples) for image classification, run
```
python build_dataset_resized_images.py big
```
To build the toy dataset (50000 training examples, 10000 test examples) for image classification, run
```
python build_dataset_resized_images.py small
```

## Quick Start
### 1. Choose hyperparameters.
Change the hyperparameters for colorization in experiments/base_model/params.json.
Change the hyperparameters for classification in experiments/base_model/image_classification_params.json.
### 2. Train your experiment. 
To train regression model of colorization, run
```
python model/main_regression.py --train [--toy if use toy dataset]
```
To train classification model of colorization, run
```
python model/main_classification.py --train [--toy if use toy dataset]
```
To train transfer learning model of colorization, run
```
python model/main_transfer_learning.py --train [--toy if use toy dataset]
```
To train image classification model, run
```
python image_colorization_model/main_vgg.py --train [--toy if use toy dataset]
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
```
python image_colorization_model/main_vgg.py --predict [--toy if use toy dataset]
```