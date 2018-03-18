import cPickle
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import sys
import os
from scipy.misc import imresize

SIZE = 100
small = (sys.argv[1] == "small")
superlarge = (sys.argv[1] == "superlarge")

DATA_TRAIN = ["data/cifar-10-batches-py/data_batch_1", 
			  "data/cifar-10-batches-py/data_batch_2",
			  "data/cifar-10-batches-py/data_batch_3",
			  "data/cifar-10-batches-py/data_batch_4",
			  "data/cifar-10-batches-py/data_batch_5"]
DATA_TEST = ["data/cifar-10-batches-py/test_batch"]
DATA_BINS = "data/bins_313.npy"

if small:
	DIR_TRAIN = "data/lab_result/" + str(SIZE) + "_train_lab/"
	DIR_TEST = "data/lab_result/" + str(SIZE) + "_test_lab/"
elif superlarge:
	DIR_TRAIN = "data/lab_result/super_train_lab/"
	DIR_TEST = "data/lab_result/super_test_lab/"
else:
	DIR_TRAIN = "data/lab_result/train_lab/"
	DIR_TEST = "data/lab_result/test_lab/"

def unpickle(file):
    with open(file, 'rb') as f:
        dict = cPickle.load(f)
    return dict

def parseImages(filenames, directory, bin_dict, bin_size):
	labels = []
	resized_images = []
	count = [0]
	
	for file in filenames:
		batch = unpickle(file)
		raw_images = batch["data"]
		batch_labels = batch["labels"]
		m, nx = raw_images.shape

		if small:
			m = SIZE

		for i in range(m):
			img = raw_images[i].reshape(3, 32, 32).transpose([1, 2, 0])

			def seperate(img):
				resized_image = imresize(img, (224, 224))
				labels.append(batch_labels[i])
				resized_images.append(resized_image)
				count[0] += 1
				print(count[0])

			if not superlarge:
				seperate(img)
			else:
				seperate(img)
				seperate(np.fliplr(img))
				# seperate(np.flipud(img))
			
	labels = np.array(labels).reshape(len(labels), -1)

	if not(os.path.exists(directory)):
		os.makedirs(directory)
	np.save(directory + "labels", labels)
	np.save(directory + "resized_images", resized_images)

	
bin_size = 10
bin_dict = np.load(DATA_BINS).tolist()
parseImages(DATA_TRAIN, DIR_TRAIN, bin_dict, bin_size)
superlarge = False
parseImages(DATA_TEST, DIR_TEST, bin_dict, bin_size)
print("Done building dataset")

'''
if small:
	assert(np.load(DIR_TRAIN + "L.npy").shape == (500, 32 ,32, 1))
	assert(np.load(DIR_TRAIN + "ab.npy").shape == (500, 32 ,32, 2))
	assert(np.load(DIR_TRAIN + "labels.npy").shape == (500, 1))
	assert(np.load(DIR_TRAIN + "bins.npy").shape == (500,32, 32, 1))
	assert(np.load(DIR_TEST + "L.npy").shape == (100, 32 ,32,1))
	assert(np.load(DIR_TEST + "ab.npy").shape == (100, 32 ,32,2))
	assert(np.load(DIR_TEST + "labels.npy").shape == (100, 1))
	assert(np.load(DIR_TEST + "bins.npy").shape == (100, 32, 32, 1))
else:
	assert(np.load(DIR_TRAIN + "L.npy").shape == (50000, 32 ,32, 1))
	assert(np.load(DIR_TRAIN + "ab.npy").shape == (50000, 32 ,32, 2))
	assert(np.load(DIR_TRAIN + "labels.npy").shape == (50000, 1))
	assert(np.load(DIR_TRAIN + "bins.npy").shape == (50000, 32, 32, 1))
	assert(np.load(DIR_TEST + "L.npy").shape == (10000, 32 ,32, 1))
	assert(np.load(DIR_TEST + "a.npy").shape == (10000, 32 ,32, 2))
	assert(np.load(DIR_TEST + "labels.npy").shape == (10000, 1))
	assert(np.load(DIR_TEST + "bins.npy").shape == (10000,32, 32, 1))
'''