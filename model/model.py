import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import Params, bins2ab, plotLabImage
import os

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
            out = tf.layers.conv2d(out, params.num_bins, 1, padding='same') 
            
        assert out.get_shape().as_list() == [None, params.image_size, params.image_size, params.num_bins]
        return out

class model:
    def __init__(self, params, arch, weights = None):
        self.params = params
        self.weights = weights
        self.arch = arch

    def build_pipeline(self, is_training, X, Y, params):
        num_samples = X.shape[0]
        if is_training:
            dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(Y)))
                .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
                .batch(params.batch_size)
                .prefetch(1)  # make sure you always have one batch ready to serve
            )
        else:
            dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(Y)))
                .batch(params.batch_size)
                .prefetch(1)  # make sure you always have one batch ready to serve
            )

        # Create reinitializable iterator from dataset
        iterator = dataset.make_initializable_iterator()
        minibatch_X, minibatch_Y = iterator.get_next()
        iterator_init_op = iterator.initializer
        pipeline = {'minibatch_X': minibatch_X, 'minibatch_Y':minibatch_Y, 'iterator_init_op': iterator_init_op}
        
        return pipeline

    def compute_cost(self, logits, labels):
        cost = tf.losses.sparse_softmax_cross_entropy(logits = logits, labels = labels)
        return cost

    def train(self, X_train, Y_train, model_dir, restore_from = None, print_cost = True):
        m = X_train.shape[0]
        pipeline = self.build_pipeline(True, X_train, Y_train, self.params)
        minibatch_X = pipeline["minibatch_X"]
        minibatch_Y = pipeline["minibatch_Y"]

        with tf.variable_scope('model', reuse = tf.AUTO_REUSE):
            arch = self.arch(minibatch_X, self.params, is_training = True)

        cost = self.compute_cost(arch.logits, minibatch_Y)
        optimizer = tf.train.AdamOptimizer(self.params.learning_rate).minimize(cost)

        last_saver = tf.train.Saver(max_to_keep = 1)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            if restore_from is not None:
                if os.path.isdir(restore_from):
                    restore_from = tf.train.latest_checkpoint(restore_from)
                    begin_at_epoch = int(restore_from.split('-')[-1])
                last_saver.restore(sess, restore_from)

            costs = [] 
            for epoch in range(self.params.num_epochs):
                print "epoch: ", epoch
                minibatch_cost = 0.
                num_minibatches = (m + self.params.batch_size - 1) // self.params.batch_size
                sess.run(pipeline['iterator_init_op'])
                
                for i in range(num_minibatches):
                    _ , temp_cost = sess.run([optimizer, cost])
                    
                    minibatch_cost += temp_cost / num_minibatches
                    print "temp_cost:", temp_cost

                # Print the cost every epoch
                if print_cost == True and epoch % 1 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                if print_cost == True and epoch % 1 == 0:
                    costs.append(minibatch_cost)

            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step = epoch + 1)

    def predict(self, X_test, Y_test, data_ab, params, restore_from):
        m = X_test.shape[0]
        pipeline = self.build_pipeline(False, X_test, Y_test, self.params)
        minibatch_X = pipeline["minibatch_X"]
        minibatch_Y = pipeline["minibatch_Y"]

        with tf.variable_scope('model', reuse = tf.AUTO_REUSE):
            arch = self.arch(minibatch_X, self.params, is_training = True)
            logits = arch.logits

        last_saver = tf.train.Saver(max_to_keep = 1)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
            last_saver.restore(sess, restore_from)

            num_minibatches = (m + self.params.batch_size - 1) // self.params.batch_size
            sess.run(pipeline['iterator_init_op'])
                    
            for i in range(num_minibatches):
                predict_logits = sess.run(logits)
                predict_bins = np.argmax(predict_logits, axis = -1)

                predict_bins = predict_bins.reshape(predict_bins.shape[0], predict_bins.shape[1], predict_bins.shape[2], 1)
                predict_ab = bins2ab(predict_bins)

            plotLabImage(X_test[0], predict_ab[0], (2, 1, 1))
            plotLabImage(X_test[0], data_ab[0], (2, 1, 2))
            plt.show()


DIR_TRAIN = "../data/lab_result/100_train_lab/"
data_L = np.load(DIR_TRAIN + "L.npy")
data_ab = np.load(DIR_TRAIN + "ab.npy")
ab_bins = np.load(DIR_TRAIN + "bins.npy")
params = Params("../experiments/base_model/params.json")


model_dir = "./weights"
save_path = os.path.join(model_dir, 'last_weights')

model = model(params, classification_8layers)
# model.train(data_L[0:100], ab_bins[0:100], model_dir)
model.predict(data_L[1:2], ab_bins[1:2], data_ab[1:2], params, save_path)



'''
pipeline = m.build_pipeline(True, data_L, ab_bins, params)
data_L = tf.placeholder(tf.float32, [None, params.image_size, params.image_size, 1])
arch = classification_8layers(data_L, params)

'''






