"""Define the model."""

import tensorflow as tf


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']
    
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 1]

    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> relu -> 3x3 conv -> relu -> batch norm
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum

    channels = [64, 128, 256, 512, 512, 512, 512, 256]
    num_convs = [2, 2, 3, 3, 3, 3, 3, 3]
    strides = [[1, 2], [1, 2], [1, 1, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 1, 1]]
    dilation = [1, 1, 1, 1, 2, 2, 1, 1]

    # padding = [1, 1, 1, 1, 2, 2, 1, 1] all same
    ''' strides
    conv1: 1 2
    conv2: 1 2
    conv3: 1 1 2
    conv4 (all dilation 1): 1 1 1
    conv5 (all dilation 2): 1 1 1    pad 2
    conv6 (dilation 2): 1 1 1     pad 2
    conv7 (dilation 1): 1 1 1 pad 1
    conv8 (dilation 1): 2 1 1 pad 1
    '''
    for i, c in enumerate(channels[0:-1]): #exclude layer 8
        with tf.variable_scope('block_{}'.format(i+1)):
            for j in range(num_convs[i]):
                s = strides[i][j]
                d = dilation[i]
                out = tf.layers.conv2d(out, c, 3, padding='same', strides = (s, s), dilation_rate = (d, d))
                out = tf.nn.relu(out)
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)

    assert out.get_shape().as_list() == [None, params.image_size / 8, params.image_size / 8, 512]

    i = len(channels) - 1
    c = channels[i]
    with tf.variable_scope('block_{}'.format(i+1)):
        # Deconvolution filter
        s = strides[i][0]
        out = tf.layers.conv2d_transpose(out, c, 4, padding = 'same', strides = (s, s))
        out = tf.layers.conv2d_transpose(out, c, 4, padding = 'same', strides = (s, s))
        out = tf.layers.conv2d_transpose(out, c, 4, padding = 'same', strides = (s, s))

        for j in range(1, num_convs[i]):
            s = strides[i][j]
            d = dilation[i]
            out = tf.layers.conv2d(out, c, 3, padding='same', strides = (s, s), dilation_rate = (d, d))
            out = tf.nn.relu(out)

    assert out.get_shape().as_list() == [None, params.image_size , params.image_size , 256]

    with tf.variable_scope('fc_1'):
        out = tf.layers.conv2d(out, params.num_bins, 1, padding='same')

    assert out.get_shape().as_list() == [None, params.image_size, params.image_size, params.num_bins]
    return out 
    '''
    out = tf.contrib.layers.flatten(out)
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, params.image_size * params.image_size * 2)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
    with tf.variable_scope('fc_2'):
        logits = tf.layers.dense(out, params.image_size * params.image_size * 2)
        out = tf.nn.sigmoid(out)
    return logits #Z
    '''

def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    assert labels.get_shape().as_list() == [None, params.image_size, params.image_size, 1]
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    # loss = tf.losses.mean_squared_error(labels=labels, predictions =logits)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    '''
    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)
    '''

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    # model_spec['accuracy'] = accuracy
    '''
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    '''
    if is_training:
        model_spec['train_op'] = train_op

    return model_spec



