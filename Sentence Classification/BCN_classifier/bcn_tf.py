# References:
# https://github.com/markmenezes11/COMPM091/blob/master/CoVe-BCN/model.py
# https://github.com/adi2103/AML-CoVe/blob/master/Classification%20Networks/BCN.ipynb 


import sys
import os
import timeit
import gc

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Bidirectional,Dropout, Embedding, \
LSTM, Multiply, Lambda, Permute, Reshape, Masking, Input, Softmax, Subtract, \
Concatenate,Dropout,MaxPooling1D,AveragePooling1D,BatchNormalization, Maximum 
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant, RandomUniform
from tensorflow.contrib.layers import maxout
from tensorflow.keras.backend import variable
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class BCN:
    def __init__(self, params, n_classes, max_sent_len, embed_dim, outputdir, weight_init=0.01, bias_init=0.01):
        self.params = params
        self.n_classes = n_classes
        self.max_sent_len = max_sent_len
        self.embed_dim = embed_dim
        self.outputdir = outputdir
        self.W_init = weight_init
        self.b_init = bias_init

    def create_model(self):
        
        print("\nCreating BCN model...")

        # Takes 2 input sequences, each of the form [GloVe(w); CoVe(w)] (duplicated if only one input sequence is needed)
        
        inputs1 = Input(shape=(max_sent_len, embed_dim, ), dtype = 'float32', batch_size = None)
        inputs2 = Input(shape=(max_sent_len, embed_dim, ), dtype = 'float32', batch_size = None)
        
        labels = Input(batch_size = None, dtype = 'int32')

        is_training = variable(dtype = 'bool')

        w_x_drop = Dropout(rate=self.params['dropout_ratio'], noise_shape=(None, 1, max_sent_len, embed_dim))(inputs1)
        w_y_drop = Dropout(rate=self.params['dropout_ratio'], noise_shape=(None, 1, max_sent_len, embed_dim))(inputs2)

        dense_layer = Dense(units=300,activation="relu")
        
        relu_x = dense_layer(w_x_drop)
        relu_y = dense_layer(w_y_drop)

        layer_lstm = Bidirectional(LSTM(units=300, return_sequences=True, activation='sigmoid'))   
        
        X = layer_lstm(relu_x)
        Y = layer_lstm(relu_y)

        biattention_input = (X,Y)


        # Biattention mechanism [Seo et al., 2017, Xiong et al., 2017]
        def biattention(biattention_input):
            X = biattention_input[0]
            Y = biattention_input[1]

            # Affinity matrix A=XY^T
            A = tf.linalg.matmul(X, Y, transpose_b=True)
            

            # Column-wise normalisation to extract attention weights
            # equation 9
            A_x = Softmax(axis=-1)(A)
            A_t = Permute((2, 1))(A)
            A_y = Softmax(axis=-1)(A_t)

            # Context summaries
            # equation 10
            C_x = tf.linalg.matmul(A_x, X, transpose_a=True)
            C_y = tf.linalg.matmul(A_y, X, transpose_a=True)
            
            # equation 11
            # X|y
            x_concat = Concatenate(axis=2)([X, X - C_y, tf.math.multiply(X, C_y)])
            x_mask = Masking(mask_value = 0.0)(x_concat)
            X_y = Bidirectional(LSTM(units=300, return_sequences = True))(x_mask)
            # equation 12
            # Y|x
            y_concat = Concatenate(axis=2)([Y, Y - C_x, tf.math.multiply(Y, C_x)])
            y_mask = Masking(mask_value = 0.0)(y_concat)
            Y_x = Bidirectional(LSTM(units=300, return_sequences = True))(y_mask)
            return X_y, Y_x

        X_y, Y_x = biattention(biattention_input)

        pool_input = (X_y,Y_x)

        # Equation 13
        def pool(pool_input):
            Xy = pool_input[0]
            Yx = pool_input[1]

            X_y_d = Dropout(rate=self.params['dropout_ratio'])(X_y)
            Y_x_d = Dropout(rate=self.params['dropout_ratio'])(Y_x)

            B_x = Dense(units = 1, activation='softmax')(X_y_d)
            B_y = Dense(units = 1, activation='softmax')(Y_x_d)
            
            # Max, mean, min and self-attentive pooling
            x_self = tf.linalg.matmul(X_y, B_x, transpose_a=True)
            y_self = tf.linalg.matmul(Y_x, B_y, transpose_a=True)

            x_self_n = tf.keras.backend.squeeze(x_self, axis=2)
            y_self_n = tf.keras.backend.squeeze(y_self, axis=2)

            x_max_pool =  tf.reduce_max(X_y, axis=1)
            x_mean_pool = tf.reduce_mean(X_y, axis=1)
            x_min_pool =  tf.reduce_min(X_y, axis=1)

            y_max_pool =  tf.reduce_max(Y_x, axis=1)
            y_mean_pool = tf.reduce_mean(Y_x, axis=1)
            y_min_pool =  tf.reduce_min(Y_x, axis=1)

            x_pool = Concatenate(axis= -1)([x_max_pool, x_mean_pool, x_min_pool, x_self_n])
            y_pool = Concatenate(axis =-1)([y_max_pool, y_mean_pool, y_min_pool, y_self_n])

        # Maxout network (2 batch-normalised maxout layers, followed by a softmax)
        with tf.variable_scope("maxout"):
            output_len = 1200
            n_classes = None ## No. of classes
            xy = Concatenate(axis =1)([x_pool, y_pool])
            result_1_dropout = Dropout(rate=self.params['dropout_ratio'])(xy)
            result_1_dense = Dense(output_len)(result_1_dropout)
            result_1_norm = BatchNormalization()(result_1_dense)
            result_1 = maxout(num_units = output_len, inputs = result_1_norm)

            result_2_dropout = Dropout(rate=self.params['dropout_ratio'])(result_1)
            result_2_dense = Dense(int(output_len/2))(result_2_dropout)
            result_2_norm = BatchNormalization()(result_2_dense)
            result_2 = maxout(num_units = output_len, inputs = result_2_norm)

            result_3_dropout = Dropout(rate=self.params['dropout_ratio'])(result_2)
            result_3_dense = Dense(N_TARGET)(result_3_dropout)
            result = tf.keras.layers.Softmax()(result_3_dense)

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=labels)

            cost = tf.reduce_mean(cross_entropy)

            if self.params['optimizer'] == "adam":
                train_step = tf.train.AdamOptimizer(self.params['learning_rate'],
                                                    beta1=self.params['adam_beta1'],
                                                    beta2=self.params['adam_beta2'],
                                                    epsilon=self.params['adam_epsilon']).minimize(cost)
            elif self.params['optimizer'] == "gradientdescent":
                train_step = tf.train.GradientDescentOptimizer(self.params['learning_rate']).minimize(cost)
            else:
                print("ERROR: Invalid optimizer: \"" + self.params['optimizer'] + "\".")
                sys.exit(1)

            predict = tf.argmax(tf.nn.softmax(logits), axis=1)

        print("BCN Model Created")
        return inputs1, inputs2, labels, is_training, predict, cost, train_step

    def dry_run(self):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            return self.create_model()

    def train(self, dataset):
        best_dev_accuracy = -1
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            inputs1, inputs2, labels, is_training, predict, loss_op, train_op = self.create_model()
        with tf.Session(graph=graph) as sess:
            print("\nTraining model...")
            sess.run(tf.global_variables_initializer())
            train_data_len = dataset.get_total_samples("train")
            total_train_batches = train_data_len // self.params['batch_size']
            train_milestones = {int(total_train_batches * 0.1): "10%", int(total_train_batches * 0.2): "20%",
                                int(total_train_batches * 0.3): "30%", int(total_train_batches * 0.4): "40%",
                                int(total_train_batches * 0.5): "50%", int(total_train_batches * 0.6): "60%",
                                int(total_train_batches * 0.7): "70%", int(total_train_batches * 0.8): "80%",
                                int(total_train_batches * 0.9): "90%", total_train_batches: "100%"}
            best_epoch_number = 0
            epochs_since_last_save = 0
            for epoch in range(self.params['n_epochs']):
                print("  ============== Epoch " + str(epoch + 1) + " of " + str(self.params['n_epochs']) + " ==============")
                epoch_start_time = timeit.default_timer()
                done = 0
                average_loss = 0
                indexes = np.random.permutation(train_data_len)
                for i in range(total_train_batches):
                    batch_indexes = indexes[i * self.params['batch_size']: (i + 1) * self.params['batch_size']]
                    batch_X1, batch_X2, batch_y = dataset.get_batch('train', batch_indexes)
                    _, loss = sess.run([train_op, loss_op], feed_dict={inputs1: batch_X1, inputs2: batch_X2,
                                                                       labels: batch_y, is_training: True})
                    average_loss += (loss / total_train_batches)
                    done += 1
                    if done in train_milestones:
                        print("    " + train_milestones[done])
                print("    Loss: " + str(average_loss))
                #print("    Computing train accuracy...")
                #train_accuracy = self.calculate_accuracy(dataset, sess, inputs1, inputs2, labels, is_training, predict, set_name="train_cut")
                #print("      Train accuracy:" + str(train_accuracy))
                print("    Computing dev accuracy...")
                dev_accuracy = self.calculate_accuracy(dataset, sess, inputs1, inputs2, labels, is_training, predict, set_name="dev")
                print("      Dev accuracy:" + str(dev_accuracy))
                print("    Epoch took %s seconds" % (timeit.default_timer() - epoch_start_time))
                if dev_accuracy > best_dev_accuracy:
                    # If dev accuracy improved, save the model after this epoch
                    best_dev_accuracy = dev_accuracy
                    best_epoch_number = epoch
                    epochs_since_last_save = 0
                    tf.train.Saver().save(sess, os.path.join(self.outputdir, 'model'))
                else:
                    # If dev accuracy got worse, don't save
                    epochs_since_last_save += 1
                gc.collect()
                if epochs_since_last_save >= 7:
                    # If dev accuracy keeps getting worse, stop training (early stopping)
                    break
            print("Finished training model after " + str(best_epoch_number + 1) + " epochs. Model is saved in: " + self.outputdir)
            print("Best dev accuracy: " + str(best_dev_accuracy))
        return best_dev_accuracy

    def calculate_accuracy(self, dataset, sess, inputs1, inputs2, labels, is_training, predict, set_name="test", verbose=False):
        test_data_len = dataset.get_total_samples(set_name)
        total_test_batches = test_data_len // self.params['batch_size']
        test_milestones = {int(total_test_batches * 0.1): "10%", int(total_test_batches * 0.2): "20%",
                           int(total_test_batches * 0.3): "30%", int(total_test_batches * 0.4): "40%",
                           int(total_test_batches * 0.5): "50%", int(total_test_batches * 0.6): "60%",
                           int(total_test_batches * 0.7): "70%", int(total_test_batches * 0.8): "80%",
                           int(total_test_batches * 0.9): "90%", total_test_batches: "100%"}
        done = 0
        test_y = []
        predicted = []
        indexes = np.arange(test_data_len)
        for i in range(total_test_batches):
            batch_indexes = indexes[i * self.params['batch_size']: (i + 1) * self.params['batch_size']]
            batch_X1, batch_X2, batch_y = dataset.get_batch(set_name, batch_indexes)
            for item in batch_y:
                test_y.append(item)
            batch_pred = list(sess.run(predict, feed_dict={inputs1: batch_X1, inputs2: batch_X2,
                                                           labels: batch_y, is_training: False}))
            for item in batch_pred:
                predicted.append(item)
            done += 1
            if verbose and done in test_milestones:
                print("  " + test_milestones[done])
        return sum([p == a for p, a in zip(predicted, test_y)]) / float(test_data_len)

    def test(self, dataset):
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            inputs1, inputs2, labels, is_training, predict, _, _ = self.create_model()
        with tf.Session(graph=graph) as sess:
            print("\nComputing test accuracy...")
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess, os.path.join(self.outputdir, 'model'))
            accuracy = self.calculate_accuracy(dataset, sess, inputs1, inputs2, labels, is_training, predict, verbose=True)
            print("Test accuracy:    " + str(accuracy))
        return accuracy
