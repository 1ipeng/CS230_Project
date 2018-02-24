from input_fn import input_fn
from model_fn import model_fn,build_model
from utils import Params
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color

DIR_TRAIN = "../data/lab_result/100_train_lab/"
data_L = np.load(DIR_TRAIN + "L.npy")
data_ab = np.load(DIR_TRAIN + "ab.npy")
params = Params("../experiments/base_model/params.json")

params.train_size = 500
train_inputs = input_fn(True, data_L, data_ab, params)
model_spec = model_fn('train', train_inputs, params)

losses=[]

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

	test_L = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
	test_ab = data_ab[0]
	test_input = {"images":test_L}
	with tf.variable_scope('model', reuse = True):
		predict_op = build_model(False, test_input, params)
	test_predict = sess.run(predict_op, feed_dict = {test_L: data_L[0].reshape(1,32,32,1)})

	t = (test_ab + 128) / 255.
	error = np.linalg.norm(test_predict[0] - t) 
	print(error)

	test_predict = test_predict[0] * 255 - 128

	plt.subplot("211")
	orig_image = np.concatenate((data_L[0], test_ab), axis = 2)
	plt.imshow(color.lab2rgb(orig_image))

	plt.subplot("212")
	pred_image = np.concatenate((data_L[0], test_predict), axis = 2)
	plt.imshow(color.lab2rgb(pred_image))
	plt.show()

	print("data_ab",data_ab[0,:5,:5])
	print("test_ab",test_predict[:5,:5])

plt.plot(losses)
plt.xlabel('Iteration i')
plt.ylabel('loss')
plt.show()



