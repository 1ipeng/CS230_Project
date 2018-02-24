"""Create the input data pipeline using `tf.data`"""
import tensorflow as tf
import numpy as np

label_dict = {}
count = 0
bin_size=10
for a in np.arange(-110, 120, bin_size):
	for b in np.arange(-110, 120, bin_size):
		label_dict[(a,b)] = count
		count += 1

def ab2label(ab,label_dict):
	m, s1, s2, c = ab.shape
	labels = np.zeros((m, s1, s2, 1))
	for i in range(m):
		for j in range(s1):
			for k in range(s2):
				a, b = ab[i, j, k, :] // bin_size * bin_size
				labels[i, j, k, 0] = label_dict[(a, b)]
	return labels


def input_fn(is_training, data_L, bins_ab, params):
	data_L = data_L / 100.
	num_samples = data_L.shape[0]

	if is_training:
		dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(data_L), tf.constant(bins_ab)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
	else:
		dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(data_L), tf.constant(bins_ab)))
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
	iterator = dataset.make_initializable_iterator()
	images, labels = iterator.get_next()
	iterator_init_op = iterator.initializer
	images = tf.cast(images, tf.float32)
	inputs = {'images': images, 'labels':labels, 'iterator_init_op': iterator_init_op}
	return inputs

'''
DIR_TRAIN = "/Users/apple/Desktop/CS230/Project/Data/small_Lab/test_lab/"
data_L = np.load(DIR_TRAIN + "L.npy")
data_a = np.load(DIR_TRAIN + "a.npy")
data_b = np.load(DIR_TRAIN + "b.npy")
inputs = input_fn(True, data_L, data_a, data_b, 10)
init_op = inputs["iterator_init_op"]
L = inputs["L"]

with tf.Session() as sess:
	sess.run(init_op)
	print(sess.run(L).shape)
dataset = tf.data.Dataset.from_tensor_slices(( np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])))
iterator = dataset.make_initializable_iterator()
a, b = iterator.get_next()
init_op = iterator.initializer

with tf.Session() as sess:
	sess.run(iterator.initializer)
	print(sess.run(b))
'''





