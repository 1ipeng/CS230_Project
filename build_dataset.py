import cPickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import color
import sys

SIZE = 100
small = (sys.argv[1] == "small")

DATA_TRAIN = ["/data/cifar-10-batches-py/data_batch_1", 
			  "/data/cifar-10-batches-py/data_batch_2",
			  "/data/cifar-10-batches-py/data_batch_3",
			  "/data/cifar-10-batches-py/data_batch_4",
			  "/data/cifar-10-batches-py/data_batch_5"]
DATA_TEST = ["/data/cifar-10-batches-py/test_batch"]

if small:
	DIR_TRAIN = "/data/lab_result/" + str(SIZE) + "_train_lab/"
	DIR_TEST = "/data/lab_result/" + str(SIZE) + "_test_lab/"
else:
	DIR_TRAIN = "/data/lab_result/train_lab/"
	DIR_TEST = "/data/lab_result/test_lab"

def unpickle(file):
    with open(file, 'rb') as f:
        dict = cPickle.load(f)
    return dict

def parseImages(filenames, directory):
	channel_L = []
	channel_ab = []
	labels = []
	for file in filenames:
		batch = unpickle(file)
		raw_images = batch["data"]
		batch_labels = batch["labels"]
		m, nx = raw_images.shape

		if small:
			m = SIZE

		for i in range(m):
			img = raw_images[i].reshape(3, 32, 32).transpose([1, 2, 0])
			Lab = color.rgb2lab(img)
			L = Lab[:, :, 0]
			L = L.reshape(L.shape[0], L.shape[1], 1)
			ab = Lab[:, :, 1:3]
			channel_L.append(L)
			channel_ab.append(ab)
			labels.append(batch_labels[i])

	labels = np.array(labels).reshape(len(labels), -1)

	return channel_L, channel_ab, labels

def ab2bins(ab):
	bin_dict = {}
	ab_dict = {}
	count = 0
	bin_size=10
	for a in np.arange(-110, 120, bin_size):
		for b in np.arange(-110, 120, bin_size):
			bin_dict[(a,b)] = count
			ab_dict[count] = [a, b]
			count += 1
	ab = np.array(ab)		
	m, s1, s2, c = ab.shape
	bins = np.zeros((m, s1, s2, 1)).astype(int)
	for i in range(m):
		for j in range(s1):
			for k in range(s2):
				a, b = ab[i, j, k, :] // bin_size * bin_size
				bins[i, j, k, 0] = bin_dict[(a, b)]
	return bins

train_L, train_ab, train_labels = parseImages(DATA_TRAIN, DIR_TRAIN)
test_L, test_ab, test_labels = parseImages(DATA_TEST, DIR_TEST)
train_bins = ab2bins(train_ab)
test_bins = ab2bins(test_ab)



np.save(DIR_TRAIN + "L", train_L)
np.save(DIR_TRAIN + "ab", train_ab)
np.save(DIR_TRAIN + "labels", train_labels)
np.save(DIR_TRAIN + "bins", train_bins)
np.save(DIR_TEST + "L", test_L)
np.save(DIR_TEST + "ab", test_ab)
np.save(DIR_TEST + "labels", test_labels)
np.save(DIR_TEST + "bins", test_bins)

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



