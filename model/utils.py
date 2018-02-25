"""General utility functions"""

import json
import logging
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
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

DATA_BINS = "../data/bins_529.npy"
bin_dict = np.load(DATA_BINS).tolist()
bins = np.random.randint(0, 50, (2,3,3,1))

def bins2ab(bins):
    m, h, w, c = bins.shape
    ab = np.zeros((m, h, w, 2))
    for i in range(m):
        for j in range(h):
            for k in range(w):
                ab[i, j, k, :] = bin_dict[bins[i, j, k, 0]]
    return ab

def ab2bins(ab):
    vfun = np.vectorize(lambda a, b: bin_dict.index([a // bin_size * bin_size, b // bin_size * bin_size]))
    bins = vfun(ab[:, :, 0], ab[:, :, 1])
    bins = bins.reshape(bins.shape[0], bins.shape[1], 1)
    return bins

def plotLabImage(L, ab, position):
    image = np.concatenate((L, ab), axis = -1)
    r, w, i = position
    plt.subplot(r, w, i)
    plt.imshow(color.lab2rgb(image))



