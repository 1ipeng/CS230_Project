import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import Params, bins2ab, plotLabImage, random_mini_batches
import os
from scipy.misc import imsave

class classification_8layers:
    def __init__(self, data_L, params, is_training=True):
        self.params = params
        self.is_training = is_training
        self.activation = {}

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
            out = tf.layers.conv2d(out, 2, 1, padding='same') 
        out = tf.sigmoid(out)
        assert out.get_shape().as_list() == [None, params.image_size, params.image_size, 2]
        return out

class model:
    def __init__(self, params, arch):
        self.params = params
        self.arch = arch
        self.X, self.Y = self.create_placeholders(self.params.image_size, self.params.image_size, 1, 2)

    def create_placeholders(self, n_H0, n_W0, n_C0, n_y):
        X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0))
        Y = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_y))
        return X, Y

    def build_architecture(self, is_training):
        with tf.variable_scope('model', reuse = tf.AUTO_REUSE):
            arch = self.arch(self.X, self.params, is_training = is_training)
        return arch

    def compute_cost(self, logits, labels):
        flat_labels = tf.contrib.layers.flatten(labels)
        flat_logits = tf.contrib.layers.flatten(logits)
        cost = tf.reduce_mean(tf.norm((flat_labels - flat_logits), axis = 1, keepdims=True) ** 2)
        return cost

    def restoreSession(self, last_saver, sess, restore_from, is_training):
        begin_at_epoch = 0
        costs = []
        dev_costs = []
        if restore_from is not None:
            if os.path.isdir(restore_from):
                sess_path = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(sess_path.split('-')[-1])
            last_saver.restore(sess, sess_path)
            
            if is_training:
                costs = np.load(os.path.join(restore_from, "costs.npy")).tolist()
                dev_costs = costs = np.load(os.path.join(restore_from, "dev_costs.npy")).tolist()

        return begin_at_epoch, costs, dev_costs

    def train(self, X_train, Y_train, X_dev, Y_dev, model_dir, restore_from = None, print_cost = True):
        m = X_train.shape[0]
        Y_train = (Y_train + 128) / 255.
        Y_dev = (Y_dev + 128) / 255.

        arch = self.build_architecture(is_training = True)
        cost = self.compute_cost(arch.logits, self.Y)
        if params.use_batch_norm:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.AdamOptimizer(self.params.learning_rate).minimize(cost)
        else:
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate).minimize(cost)

        last_saver = tf.train.Saver(max_to_keep = 1)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            begin_at_epoch, costs, dev_costs = self.restoreSession(last_saver, sess, restore_from, is_training = True)
            
            for epoch in range(self.params.num_epochs):
                print "epoch: ", epoch
                minibatch_cost = 0.
                num_minibatches = (m + self.params.batch_size - 1) // self.params.batch_size

                minibatches = random_mini_batches(X_train, Y_train, self.params.batch_size)
 
                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    _ , temp_cost = sess.run([optimizer, cost], feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})
                    
                    minibatch_cost += temp_cost / num_minibatches
                
                costs.append(minibatch_cost) 
                dev_cost = self.evaluate(X_dev, Y_dev, self.params, sess)
                dev_costs.append(dev_cost)

                if print_cost == True and epoch % 1 == 0:
                    print ("Cost after epoch %i: %f" % (begin_at_epoch + epoch, minibatch_cost))          
                    print ("dev_Cost after epoch %i: %f" % (begin_at_epoch + epoch, dev_cost))   

            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step = begin_at_epoch + epoch + 1)
            np.save(os.path.join(model_dir,'last_weights', "costs"), costs)
            np.save(os.path.join(model_dir,'last_weights', "dev_costs"), dev_costs)  

    def evaluate(self, X_test, Y_test, params, sess):
        m = X_test.shape[0]
        arch = self.build_architecture(is_training = False)
        
        logits = arch.logits
        cost = self.compute_cost(arch.logits, self.Y)                
        predict_cost = sess.run(cost, feed_dict={self.X: X_test, self.Y: Y_test})
        return predict_cost

    def predict(self, X_test, data_ab, params, restore_from):
        m = X_test.shape[0]
        Y_test = (data_ab + 128) / 255.

        arch = self.build_architecture(is_training = False)
        
        logits = arch.logits
        cost = self.compute_cost(arch.logits, self.Y)
        last_saver = tf.train.Saver(max_to_keep = 1)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            self.restoreSession(last_saver, sess, restore_from, False)
                    
            predict_cost, predict_ab = sess.run([cost, logits], feed_dict={self.X: X_test, self.Y: Y_test})
            predict_ab = predict_ab * 255. - 128
            
            predict_img = plotLabImage(X_test[0], predict_ab[0], (2, 1, 1))
            orig_img = plotLabImage(X_test[0], data_ab[0], (2, 1, 2))
            plt.show()
            
        return predict_ab, predict_cost
                
DIR_TRAIN = "../data/lab_result/100_train_lab/"
data_L = np.load(DIR_TRAIN + "L.npy")
data_ab = np.load(DIR_TRAIN + "ab.npy")
ab_bins = np.load(DIR_TRAIN + "bins.npy")
params = Params("../experiments/base_model/params.json")

train_size = 100
dev_size = 30
m = data_L.shape[0]
permutation = list(np.random.permutation(m))

train_L = data_L[0:train_size]
train_ab = data_ab[0:train_size]
train_bins = ab_bins[0:train_size]
dev_L = data_L[train_size:train_size + dev_size]
dev_ab = data_ab[train_size:train_size + dev_size]
dev_bins = ab_bins[train_size:train_size + dev_size]

model_dir = "./weights"
save_path = os.path.join(model_dir, 'last_weights')

model = model(params, classification_8layers)
model.train(train_L[0:1], train_ab[0:1], train_L[0:1], train_ab[0:1], model_dir)

model.predict(train_L, train_ab, params, save_path)

'''
pipeline = m.build_pipeline(True, data_L, ab_bins, params)
data_L = tf.placeholder(tf.float32, [None, params.image_size, params.image_size, 1])
arch = classification_8layers(data_L, params)

'''


