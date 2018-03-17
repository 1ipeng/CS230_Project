"""General utility functions"""

import json
import logging
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import math
class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

DATA_BINS = "../data/bins_313.npy"
bin_dict = np.load(DATA_BINS).tolist()
np_bin_dict = np.load(DATA_BINS)

def bins2ab(bins):
    m, h, w, c = bins.shape
    ab = np.zeros((m, h, w, 2))
    for i in range(m):
        for j in range(h):
            for k in range(w):
                ab[i, j, k, :] = bin_dict[bins[i, j, k, 0]]
    return ab

def annealed_mean(logits, T):
    temp = np.exp(logits / T)
    s = np.sum(temp, axis = -1, keepdims = True)
    annealed_prob = temp / s
    a = np.array(np_bin_dict[:, 0])
    b = np.array(np_bin_dict[:, 1])
    annealed_mean = np.zeros((annealed_prob.shape[0], annealed_prob.shape[1], annealed_prob.shape[2], 2))
    annealed_mean_a = np.sum(a.reshape(1, 1, 1, -1) * annealed_prob, axis = -1)
    annealed_mean_b = np.sum(b.reshape(1, 1, 1, -1) * annealed_prob, axis = -1)
    annealed_mean[:,:,:, 0] = annealed_mean_a
    annealed_mean[:,:,:, 1] = annealed_mean_b
    return annealed_mean

def ab2bins(ab):
    vfun = np.vectorize(lambda a, b: bin_dict.index([a // bin_size * bin_size, b // bin_size * bin_size]))
    bins = vfun(ab[:, :, 0], ab[:, :, 1])
    bins = bins.reshape(bins.shape[0], bins.shape[1], 1)
    return bins

def plotLabImage(L, ab, position, grayScale = False):
    r, w, i = position
    image = np.concatenate((L, ab), axis = -1)
    plt.subplot(r, w, i)
    RGB = color.lab2rgb(image)
    if grayScale:
        RGB = color.rgb2gray(RGB)
        plt.imshow(RGB, cmap = "gray")
    else:
        plt.imshow(RGB)
    return RGB

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches