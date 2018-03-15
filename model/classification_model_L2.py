# Classification model
import tensorflow as tf
import os

# Define architecture
class classification_8layers_model:
    def __init__(self, params, is_training=True):
        # params: hyperparameters
        self.params = params
        self.is_training = is_training
        self.activation = {}
        self.X, self.Y = self.create_placeholders()
        normalized_X = self.normalize(self.X)
        conv_out = self.convlayers(normalized_X)
        deconv_out = self.deconvlayers(conv_out)
        self.logits = self.fc_layers(deconv_out)
        self.probs = tf.nn.softmax(self.logits)
        self.cost = self.compute_cost(self.logits, self.Y)
        self.l2_cost = self.compute_l2_cost(self.logits, self.Y)

    def create_placeholders(self):
        X = tf.placeholder(tf.float32, shape = (None, self.params.image_size, self.params.image_size, 1))
        Y = tf.placeholder(tf.int32, shape = (None, self.params.image_size, self.params.image_size, 1))
        return X, Y

    def compute_l2_cost(self, logits, labels):
        # Softmax loss + L2 Loss
        l2_loss = tf.losses.get_regularization_loss()
        cost = tf.losses.sparse_softmax_cross_entropy(logits = logits, labels = labels) + l2_loss
        return cost

    def compute_cost(self, logits, labels):
        # Softmax loss
        cost = tf.losses.sparse_softmax_cross_entropy(logits = logits, labels = labels)
        return cost

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

        assert out.get_shape().as_list() == [None, self.params.image_size, self.params.image_size, 1]
    
        for i, c in enumerate(channels): # exclude layer 8 (deconvolution)
            block_name = "block_" + str(i+1)
            for j in range(num_convs[i]):
                conv_name = "conv_" + str(i+1) + "_" + str(j+1)
                with tf.variable_scope(conv_name):
                    s = strides[i][j]
                    d = dilation[i]
                    out = tf.layers.conv2d(out, c, 3, padding='same', strides = (s, s), dilation_rate = (d, d), kernel_regularizer= tf.contrib.layers.l2_regularizer(self.params.reg_constant))
                    out = tf.nn.relu(out)
                    out = tf.layers.dropout(inputs=out, rate=self.params.dropout_rate, training=is_training)

            if self.params.use_batch_norm:
                bn_name = "bn_" + str(i+1)
                with tf.variable_scope(bn_name):
                    out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)

            self.activation[block_name] = out
            self.out = out

        assert out.get_shape().as_list() == [None, self.params.image_size / 8, self.params.image_size / 8, 512]
        return out

    def deconvlayers(self, out):
        # 3 deconv layers: 4x4 deconv -> relu -> 4x4 deconv -> relu -> 4x4 deconv -> relu
        block_name = "block_8"
        s = 2 #stride
        c = 256 #channels
        for i in range(3):
             with tf.variable_scope('deconv_' + str(i+1)):
                out = tf.layers.conv2d_transpose(out, c, 4, padding = 'same', strides = (s, s), kernel_regularizer= tf.contrib.layers.l2_regularizer(self.params.reg_constant))
                out = tf.nn.relu(out)
                out = tf.layers.dropout(inputs=out, rate=self.params.dropout_rate, training=self.is_training)

        assert out.get_shape().as_list() == [None, self.params.image_size , self.params.image_size , 256]
        
        self.activation[block_name] = out
        return out

    def fc_layers(self, out):
        # 1x1 conv -> softmax
        with tf.variable_scope('fc_1'):
            out = tf.layers.conv2d(out, self.params.num_bins, 1, padding='same',kernel_regularizer= tf.contrib.layers.l2_regularizer(self.params.reg_constant)) 
            out = tf.layers.dropout(inputs=out, rate=self.params.dropout_rate, training=self.is_training)

        assert out.get_shape().as_list() == [None, self.params.image_size, self.params.image_size, self.params.num_bins]
        return out