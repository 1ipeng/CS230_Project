# Classification model

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import Params, bins2ab, plotLabImage, random_mini_batches
import os
from scipy.misc import imsave
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--restore", help="restore training from last epoch",
                    action="store_true")
parser.add_argument("--train", help="train model",
                    action="store_true")
parser.add_argument("--predict", help="show predict results",
                    action="store_true")

args = parser.parse_args()
if len(sys.argv) < 2:
    parser.print_usage()

# Define architecture
class classification_8layers:
    def __init__(self, data_L, params, is_training=True):
        # data_L: input placeholder
        # params: hyperparameters

        self.params = params
        self.is_training = is_training
        self.activation = {} # save activation for debugging

        input_L = self.normalize(data_L)
        conv_out = self.convlayers(input_L)
        deconv_out = self.deconvlayers(conv_out)

        self.logits = self.fc_layers(deconv_out)
        self.probs = tf.nn.softmax(self.logits)

    def normalize(self, data_L):
        input_L = tf.cast(data_L, tf.float32)
        input_L = input_L / tf.constant(100.)
        
        return input_L

    def convlayers(self, out):
        # Define the number of channels of each convolution block
        channels = [64, 128, 256, 512, 512, 512, 512]

        # For each block, we do: 3x3 conv -> relu -> 3x3 conv -> relu ->( 3x3 conv -> relu )-> batch norm
        # Number of convs each convolution block (2 or 3)
        num_convs = [2, 2, 3, 3, 3, 3, 3]

        # Strides for each conv
        strides = [[1, 2], [1, 2], [1, 1, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
       
        # Dilation for each conv
        dilation = [1, 1, 1, 1, 2, 2, 1]

        # Batch norm momentum
        bn_momentum = self.params.bn_momentum
        is_training = self.is_training

        assert out.get_shape().as_list() == [None, params.image_size, params.image_size, 1]
    
        for i, c in enumerate(channels): # exclude layer 8 (deconvolution)
            block_name = "block_" + str(i+1)
            for j in range(num_convs[i]):
                conv_name = "conv_" + str(i+1) + "_" + str(j+1)
                with tf.variable_scope(conv_name):
                    s = strides[i][j]
                    d = dilation[i]
                    out = tf.layers.conv2d(out, c, 3, padding='same', strides = (s, s), dilation_rate = (d, d))
                    out = tf.nn.relu(out)

            if params.use_batch_norm:
                bn_name = "bn_" + str(i+1)
                with tf.variable_scope(bn_name):
                    out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)

            self.activation[block_name] = out
            self.out = out

        assert out.get_shape().as_list() == [None, params.image_size / 8, params.image_size / 8, 512]
        return out

    def deconvlayers(self, out):
        # 3 deconv layers: 4x4 deconv -> relu -> 4x4 deconv -> relu -> 4x4 deconv -> relu
        block_name = "block_8"
        s = 2 #stride
        c = 256 #channels
        for i in range(3):
             with tf.variable_scope('deconv_' + str(i+1)):
                out = tf.layers.conv2d_transpose(out, c, 4, padding = 'same', strides = (s, s))
                out = tf.nn.relu(out)

        assert out.get_shape().as_list() == [None, params.image_size , params.image_size , 256]
        
        self.activation[block_name] = out
        return out

    def fc_layers(self, out):
        # 1x1 conv -> softmax
        with tf.variable_scope('fc_1'):
            out = tf.layers.conv2d(out, params.num_bins, 1, padding='same') 
            
        assert out.get_shape().as_list() == [None, params.image_size, params.image_size, params.num_bins]
        return out

class model:
    def __init__(self, params, arch):
        # params: hyperparameter
        # arch: Network architeture 
        self.params = params
        self.arch = arch
        self.X, self.Y = self.create_placeholders(self.params.image_size, self.params.image_size, 1, 1)

    def create_placeholders(self, n_H0, n_W0, n_C0, n_y):
        X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0))
        Y = tf.placeholder(tf.int32, shape = (None, n_H0, n_W0, n_y))
        return X, Y

    def build_architecture(self, is_training):
        with tf.variable_scope('model', reuse = tf.AUTO_REUSE):
            arch = self.arch(self.X, self.params, is_training = is_training)
        return arch

    def compute_cost(self, logits, labels):
        # Softmax loss
        cost = tf.losses.sparse_softmax_cross_entropy(logits = logits, labels = labels)
        return cost

    def restoreSession(self, last_saver, sess, restore_from, is_training):
        # Restore sess, cost from last training
        begin_at_epoch = 0
        costs = []
        dev_costs = []
        best_dev_cost = float('inf')
        if restore_from is not None:
            if os.path.isdir(restore_from):
                sess_path = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(sess_path.split('-')[-1])
            last_saver.restore(sess, sess_path)
            
            if is_training:
                costs = np.load(os.path.join(restore_from, "costs.npy")).tolist()
                dev_costs = np.load(os.path.join(restore_from, "dev_costs.npy")).tolist()
                best_dev_cost = np.load(os.path.join(restore_from,"best_dev_cost.npy"))[0]

        return begin_at_epoch, costs, dev_costs, best_dev_cost

    def train(self, X_train, Y_train, X_dev, Y_dev, model_dir, restore_from = None, print_cost = True):
        m = X_train.shape[0]
        
        arch = self.build_architecture(is_training = True)
        cost = self.compute_cost(arch.logits, self.Y)
        if params.use_batch_norm:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.AdamOptimizer(self.params.learning_rate).minimize(cost)
        else:
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate).minimize(cost)

        last_saver = tf.train.Saver(max_to_keep = 1)
        best_saver = tf.train.Saver(max_to_keep = 1)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            begin_at_epoch, costs, dev_costs, best_dev_cost = self.restoreSession(last_saver, sess, restore_from, is_training = True)
            
            for epoch in range(self.params.num_epochs):
                print ("epoch: ", epoch + 1)
                minibatch_cost = 0.
                num_minibatches = (m + self.params.batch_size - 1) // self.params.batch_size

                minibatches = random_mini_batches(X_train, Y_train, self.params.batch_size)
 
                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    _ , temp_cost = sess.run([optimizer, cost], feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})
                    
                    # compute training cost
                    minibatch_cost += temp_cost / num_minibatches
                
                costs.append(minibatch_cost) 

                # compute dev cost
                dev_cost = self.evaluate(X_dev, Y_dev, self.params, sess)
                dev_costs.append(dev_cost)

                if print_cost == True and epoch % 1 == 0:
                    print ("Cost after epoch %i: %f" % (begin_at_epoch + epoch + 1, minibatch_cost))          
                    print ("dev_Cost after epoch %i: %f" % (begin_at_epoch + epoch + 1, dev_cost))   

                # Save best sess
                if dev_cost < best_dev_cost:
                    best_dev_cost = dev_cost
                    best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                    best_saver.save(sess, best_save_path, global_step = begin_at_epoch + epoch + 1)
                    if not (os.path.exists(os.path.join(model_dir,'last_weights'))):
                        os.makedirs(os.path.join(model_dir,'last_weights'))
                    np.save(os.path.join(model_dir,'last_weights', "bes_dev_cost"), best_dev_cost)

            # Save sess and costs
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step = begin_at_epoch + epoch + 1)
            np.save(os.path.join(model_dir,'last_weights', "costs"), costs)
            np.save(os.path.join(model_dir,'last_weights', "dev_costs"), dev_costs)  

    def evaluate(self, X_test, Y_test, params, sess):
        # Evaluate the dev set. Used inside a session.
        m = X_test.shape[0]
        arch = self.build_architecture(is_training = False)
        logits = arch.logits
        cost = self.compute_cost(arch.logits, self.Y)                
        predict_cost = sess.run(cost, feed_dict={self.X: X_test, self.Y: Y_test})
        return predict_cost

    def predict(self, X_test, Y_test, data_ab, params, restore_from):
        # Make prediction. Used outside a session.
        m = X_test.shape[0]
        arch = self.build_architecture(is_training = False)
        
        logits = arch.logits
        cost = self.compute_cost(arch.logits, self.Y)
        last_saver = tf.train.Saver(max_to_keep = 1)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            self.restoreSession(last_saver, sess, restore_from, False)
                    
            predict_cost, predict_logits = sess.run([cost, logits], feed_dict={self.X: X_test, self.Y: Y_test})
            predict_bins = np.argmax(predict_logits, axis = -1)

            predict_bins = predict_bins.reshape(predict_bins.shape[0], predict_bins.shape[1], predict_bins.shape[2], 1)
            predict_ab = bins2ab(predict_bins)
            
        return predict_bins, predict_ab, predict_cost


# Experiment on a toy dataset with train size 100, dev size 30

# Load data
DIR_TRAIN = "../data/lab_result/100_train_lab/"
data_L = np.load(DIR_TRAIN + "L.npy")
data_ab = np.load(DIR_TRAIN + "ab.npy")
ab_bins = np.load(DIR_TRAIN + "bins.npy")
params = Params("../experiments/base_model/params.json")

# Shuffle data
train_size = 100
dev_size = 30
m = data_L.shape[0]
np.random.seed(10)
permutation = list(np.random.permutation(m))
train_index = permutation[0:train_size]
dev_index = permutation[train_size:train_size + dev_size]

# Build toy dataset
train_L = data_L[train_index]
train_ab = data_ab[train_index]
train_bins = ab_bins[train_index]
dev_L = data_L[dev_index]
dev_ab = data_ab[dev_index]
dev_bins = ab_bins[dev_index]

model_dir = "./weights_classification"
save_path = os.path.join(model_dir, 'last_weights')

# Build model
model = model(params, classification_8layers)


# Train and predict
if args.train:
	if args.restore:
	    model.train(train_L, train_bins, dev_L, dev_bins, model_dir, save_path)
	else:
	    model.train(train_L, train_bins, dev_L, dev_bins, model_dir)

# Show result
if args.predict:
	save_path = os.path.join(model_dir, 'last_weights')
	predict_bins, predict_ab, predict_cost = model.predict(dev_L[0:5], dev_bins[0:5], dev_ab[0:5], params, save_path)
	count = 0
	for i in range(5):
	    count = count + 1
	    orig_img = plotLabImage(dev_L[i], dev_ab[i], (5, 2, count))
	    count = count + 1
	    predict_img = plotLabImage(dev_L[i], predict_ab[i], (5, 2, count))

	plt.figure()
	predict_bins, predict_ab, predict_cost = model.predict(train_L[0:5], train_bins[0:5], train_ab[0:5], params, save_path)
	count = 0
	for i in range(5):
	    count = count + 1
	    orig_img = plotLabImage(train_L[i], train_ab[i], (5, 2, count))
	    count = count + 1
	    predict_img = plotLabImage(train_L[i], predict_ab[i], (5, 2, count))
	plt.show()