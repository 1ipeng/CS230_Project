from classification_8layers_input import input_fn
from classification_8layers_model import model_fn,build_model
from utils import Params
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color

DIR_TRAIN = "../data/lab_result/100_train_lab/"
data_L = np.load(DIR_TRAIN + "L.npy")
test_ab = np.load(DIR_TRAIN + "ab.npy")
ab_bins = np.load(DIR_TRAIN + "bins.npy")
params = Params("../experiments/base_model/params.json")

params.train_size = 1
train_inputs = input_fn(True, data_L, ab_bins, params)
model_spec = model_fn('train', train_inputs, params)

losses=[]	

ab_dict = {}
count = 0
bin_size=10
for a in np.arange(-110, 120, bin_size):
	for b in np.arange(-110, 120, bin_size):
		ab_dict[count] = [a, b]
		count += 1


def predictY(data_L, test_ab):
	test_L = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
	test_input = {"images":test_L}
	with tf.variable_scope('model', reuse = True):
		predict_op = build_model(True, test_input, params)
	test_logits = sess.run(predict_op, feed_dict = {test_L: data_L.reshape(1,32,32,1)})
	test_predict = np.argmax(test_logits, axis = -1)
	print test_predict
	_, h, w, _ = test_predict.shape

	predict_ab = np.zeros((h, w, 2))

	for i in range(h):
		for j in range(w):
			predict_ab[i, j, :] = ab_dict[test_predict[0, i, j, 0]]

	orig_image = np.concatenate((data_L, test_ab), axis = 2)
	pred_image = np.concatenate((data_L, predict_ab), axis = 2)
	error = np.linalg.norm(predict_ab - test_ab)
	print error


with tf.Session() as sess:
	sess.run(model_spec['variable_init_op'])

	for epoch in range(params.num_epochs):
		print('epoch',epoch)
		num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
		loss = model_spec['loss']
		train_op = model_spec['train_op']
		global_step = tf.train.get_global_step()
		sess.run(model_spec['iterator_init_op'])

		t = range(num_steps)
		for i in t:
			_, loss_val = sess.run([train_op, loss])
			print loss_val
			losses.append(loss_val)

	predictY(data_L[0], test_ab[0])
'''
	plt.imshow(color.lab2rgb(pred_image))
	plt.show()

	print("ab_bins",ab_bins[0,:5,:5])
	print("test_ab",test_predict[:5,:5])

plt.plot(losses)
plt.xlabel('Iteration i')
plt.ylabel('loss')
plt.show()
'''



