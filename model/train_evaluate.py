import tensorflow as tf
import numpy as np
from utils import bins2ab, random_mini_batches
import os

class train_evaluate:
    def __init__(self, params, model, weights_file = None, model_type = 'classification'):
        # params: hyperparameter
        # model: Network model 
        self.params = params
        self.weights_file = weights_file
        self.model_type = model_type
        self.train_model, self.test_model = self.build_model(model)

    def build_model(self, model):
        with tf.variable_scope('model', reuse = False):
            train_model = model(self.params, is_training = True)
        with tf.variable_scope('model', reuse = True):
            test_model = model(self.params, is_training = False)
        return train_model, test_model

    def restoreSession(self, last_saver, sess, restore_from, is_training):
        # Restore sess, cost from last training
        begin_at_epoch = 0
        costs = []
        dev_costs = []
        best_dev_accuracy = float('-inf')
        dev_accuracies = []
        if restore_from is not None:
            if os.path.isdir(restore_from):
                sess_path = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(sess_path.split('-')[-1])
            last_saver.restore(sess, sess_path)
            
            if is_training:
                costs = np.load(os.path.join(restore_from, "costs.npy")).tolist()
                dev_costs = np.load(os.path.join(restore_from, "dev_costs.npy")).tolist()
                dev_accuracies = np.load(os.path.join(restore_from, "dev_accuracies.npy")).tolist()
                best_dev_accuracy = np.load(os.path.join(restore_from,"best_dev_accuracy.npy"))[0]

        return begin_at_epoch, costs, dev_costs, best_dev_accuracy, dev_accuracies

    def train(self, X_train, Y_train, X_dev, Y_dev, model_dir, restore_from = None, print_cost = True):
        m = X_train.shape[0]
        
        model = self.train_model
        l2_cost = model.l2_cost
        accuracy = model.accuracy
        cost = model.cost

        if self.params.use_batch_norm:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.AdamOptimizer(self.params.learning_rate).minimize(l2_cost)
        else:
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate).minimize(l2_cost)

        last_saver = tf.train.Saver(max_to_keep = 1)
        best_saver = tf.train.Saver(max_to_keep = 1)
        with tf.Session() as sess:
            if self.weights_file is not None:
                model.load_weights(self.weights_file, sess)

            init = tf.global_variables_initializer()
            sess.run(init)

            begin_at_epoch, costs, dev_costs, best_dev_accuracy, dev_accuracies = self.restoreSession(last_saver, sess, restore_from, is_training = True)
            
            for epoch in range(self.params.num_epochs):
                count_batch = 0
                print ("epoch: ", epoch + 1)
                minibatch_cost = 0.
                minibatch_accuracy = 0.
                num_minibatches = (m + self.params.train_batch_size - 1) // self.params.train_batch_size

                minibatches = random_mini_batches(X_train, Y_train, self.params.train_batch_size)
 
                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    _ , temp_cost, temp_accuracy = sess.run([optimizer, cost, accuracy], feed_dict={model.X: minibatch_X, model.Y: minibatch_Y})
                    
                    # compute training cost
                    minibatch_cost += temp_cost / num_minibatches
                    minibatch_accuracy += temp_accuracy / num_minibatches

                    # Print result
                    if (count_batch % 10) == 0:
                        print("count_batch",count_batch,"temp_cost:", temp_cost, "temp_accuracy:", temp_accuracy)
                    count_batch += 1
                
                costs.append(minibatch_cost) 

                # compute dev cost
                dev_cost, dev_accuracy = self.evaluate(X_dev, Y_dev, sess)
                dev_costs.append(dev_cost)
                dev_accuracies.append(dev_accuracy)

                if print_cost == True and epoch % 1 == 0:
                    print ("Cost after epoch %i: %f" % (begin_at_epoch + epoch + 1, minibatch_cost))    
                    print ("Accuracy after epoch %i: %f" % (begin_at_epoch + epoch + 1, minibatch_accuracy))       
                    print ("dev_Cost after epoch %i: %f" % (begin_at_epoch + epoch + 1, dev_cost))
                    print ("dev_accuracy after epoch %i: %f" % (begin_at_epoch + epoch + 1, dev_accuracy))
                
                # Save best sess
                if dev_accuracy > best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy
                    best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                    best_saver.save(sess, best_save_path, global_step = begin_at_epoch + epoch + 1)
                    if not (os.path.exists(os.path.join(model_dir,'last_weights'))):
                        os.makedirs(os.path.join(model_dir,'last_weights'))
                    np.save(os.path.join(model_dir,'last_weights', "best_dev_accuracy"), [best_dev_accuracy])

            # Save sess and costs
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step = begin_at_epoch + epoch + 1)
            np.save(os.path.join(model_dir,'last_weights', "costs"), costs)
            np.save(os.path.join(model_dir,'last_weights', "dev_costs"), dev_costs)  
            np.save(os.path.join(model_dir,'last_weights', "dev_accuracies"), dev_accuracies) 

    def evaluate(self, X_test, Y_test, sess):
        # Evaluate the dev set. Used inside a session.
        m = X_test.shape[0]
        model = self.test_model
        accuracy = model.accuracy
        logits = model.logits
        cost = model.cost     

        minibatches = random_mini_batches(X_test, Y_test, self.params.test_batch_size)
        minibatch_cost = 0.
        minibatch_accuracy = 0.
        num_minibatches = (m + self.params.test_batch_size - 1) // self.params.test_batch_size

        count_batch=0
        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            temp_cost, temp_accuracy = sess.run([cost, accuracy], feed_dict={model.X: minibatch_X, model.Y: minibatch_Y})
            
            # compute dev cost
            minibatch_cost += temp_cost / num_minibatches
            minibatch_accuracy += temp_accuracy / num_minibatches

            # Print result
            #if (count_batch % 10) == 0:
            print("dev_count_batch",count_batch,"dev_temp_cost:", temp_cost, "dev_temp_accuracy:", temp_accuracy)
            count_batch += 1

        return minibatch_cost, minibatch_accuracy

    def predict(self, X_test, Y_test, restore_from):
        # Make prediction. Used outside a session.
        m = X_test.shape[0]
        model = self.test_model
        accuracy = model.accuracy
        logits = model.logits
        # probs = model.probs
        cost = model.cost   
        last_saver = tf.train.Saver(max_to_keep = 1)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.restoreSession(last_saver, sess, restore_from, False)

            predict_costs = np.zeros(m)
            predict_accuracy = np.zeros(m)
            if self.model_type == 'classification':
                predict_logits = np.zeros((m, self.params.image_size, self.params.image_size, self.params.num_bins))
            else:
                predict_logits = np.zeros((m, self.params.image_size, self.params.image_size, 2))
            # predict_probs = np.zeros((m, self.params.image_size, self.params.image_size, self.params.num_bins))
            
            for i in range(m):
                predict_costs[i], predict_logits[i, :, :, :], check, predict_accuracy[i] = sess.run([cost, logits, model.check, accuracy], feed_dict={model.X: X_test[i:i+1], model.Y: Y_test[i:i+1]})

            if self.model_type == 'classification':
                predict_bins = np.argmax(predict_logits, axis = -1)
                predict_bins = predict_bins.reshape(predict_bins.shape[0], predict_bins.shape[1], predict_bins.shape[2], 1)
                predict_ab = bins2ab(predict_bins)
                return predict_bins, predict_ab, predict_costs, predict_logits, predict_accuracy, check
            else:
                predict_ab = predict_logits * 255. - 128
                return predict_ab, predict_costs, predict_logits, predict_accuracy, check
