#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:14:08 2019

@author: oggi2
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from sklearn.metrics import f1_score

#os.environ['CUDA_VISIBLE_DEVICES']='0,1'  # Number of GPUs to run on
os.environ['CUDA_VISIBLE_DEVICES']='-1'

class Tagger(object):

    def __init__(self, model_file, n_steps, n_input, feature_number, training = True, epochs = 10, expnum = 0, cvit=None, dropout=False):
        
        self.expnum = expnum
        self.header = 'EXP_{0}/new_checkpoint'.format(self.expnum)
        self.header_test = 'EXP_{0}/new_checkpoint_test'.format(self.expnum)
        self.model_file = model_file
        if not os.path.exists(self.header + os.sep + self.model_file):
            os.makedirs(self.header + os.sep + self.model_file)

        self.name = "LSTM"
        self.training = training

        self.learning_rate = 1e-3
        self.n_batches = 10
        self.batch_size = 5
        self.display_step = 33
        self.save_step = 5
        self.epochs = epochs
        self.feature_number = feature_number
        self.acc_f = []
        self.loss_f = []
        self.cvit = cvit

        # Network Parameters

        self.n_input = n_input
        self.n_steps = n_steps
        self.n_hidden = 64
        self.n_classes = 3


        # classifier
        def lstm(x, weights, biases):
            x = tf.unstack(x, self.n_steps, 1)

            ## TODO: if dropout##
            if dropout:
                keep_prob_in = 0.25
                x = [tf.nn.dropout(x_i, keep_prob_in) for x_i in x]
            ##################################################################################
            with tf.variable_scope('tagger1_feature_{0}'.format(self.feature_number)):
                lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
            with tf.variable_scope('tagger2_feature_{0}'.format(self.feature_number)):
                self.outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                # take the self.outputs to the Q-network
            ## TODO: if dropout##
            if dropout:
                keep_prob_out = 0.5
                self.outputs = [tf.nn.dropout(rnn_output, keep_prob_out) for rnn_output in self.outputs]
            ###################################################################################
            logitx = tf.stack(self.outputs)            
            self.avg_outputs = tf.reduce_mean(tf.stack(self.outputs), 0)

            pred = tf.matmul(self.avg_outputs, weights['out']) + biases['out']
            matmul = tf.zeros([10,3],tf.int32)
            return pred, logitx, matmul
        self.x = tf.placeholder("float", [None, self.n_steps, self.n_input])
        self.y = tf.placeholder("int32", [None])

    # Define weights
        with tf.variable_scope('weight_feature_{0}'.format(self.feature_number)):
            self.weights = {'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))}
        with tf.variable_scope('bias_feature_{0}'.format(self.feature_number)):
            self.biases = {'out': tf.Variable(tf.random_normal([self.n_classes]))}

    # LSTM
    
        self.pred, self.xlogits, self.xfcls = lstm(self.x, self.weights, self.biases)
        self.sm = tf.nn.softmax(self.pred)

        # Define loss and optimizer

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        with tf.variable_scope('adam1_feature_{0}'.format(self.feature_number)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Evaluate model

        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.cast(self.y, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            
        self.sess = tf.Session(graph=tf.get_default_graph())
        self.saver = tf.train.Saver()
        
                
    def train(self, data_x, data_y, feature_number):


        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)        
        if ckpt and ckpt.model_checkpoint_path and (data_x != []):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path) 
        else:
            self.sess.run(tf.initialize_variables(tf.global_variables()))
            

        if len(data_y) >= 150:
            self.batch_size = 32

        for i in range(self.epochs):
            step = 1
            while step * self.batch_size <= len(data_y):

                batch_x = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                batch_y = data_y[(step - 1) * self.batch_size:step * self.batch_size]

                self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})

                if step % self.display_step == 0:
                    acc = self.sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                    loss = self.sess.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y})
                    #print("Epoch: " + str(i + 1) + ", iter: " + str(
                    #    step * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                    #    loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                    self.loss_f.append(loss)
                    self.acc_f.append(100 * acc)

                step += 1

            if (i+1) % self.save_step == 0:
                self.saver.save(self.sess, self.header + os.sep + self.model_file + os.sep + 'model.ckpt', i+1)

    ### used during testing        
    def train_mode_B(self, data_x, data_y, feature_number, mode = 'B'):

        if not os.path.exists(self.header_test + os.sep + 'test_{0}_{1}_/'.format(mode, self.cvit) +self.model_file):
            os.makedirs(self.header_test + os.sep + 'test_{0}_{1}_/'.format(mode, self.cvit) + self.model_file)
            
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            '###############################ERROR#############################'

        if len(data_y) >= 150:
            self.batch_size = 32
        for i in range(self.epochs):
            step = 1
            while step * self.batch_size <= len(data_y):

                batch_x = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                batch_y = data_y[(step - 1) * self.batch_size:step * self.batch_size]

                self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})

                if step % self.display_step == 0:
                    acc = self.sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                    loss = self.sess.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y})
                    #print("Epoch: " + str(i + 1) + ", iter: " + str(
                    #    step * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                    #    loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                    self.loss_f.append(loss)
                    self.acc_f.append(100 * acc)

                step += 1

            if (i+1) % self.save_step == 0:
                self.saver.save(self.sess, self.header_test + os.sep  + 'test_{0}_{1}_/'.format(mode, self.cvit) + self.model_file + os.sep + 'model.ckpt', i+1)



    def get_predictions(self, x):
        
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x: x}), 1)
        return pred

    # return shpae of 10*3
    def get_xfcls(self, x):
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        if len(np.array(x).shape)==2:
            xfcls = self.sess.run(self.xfcls, feed_dict={self.x: [x]})
        else:
            xfcls = self.sess.run(self.xfcls, feed_dict={self.x: x})
        return xfcls
    
    # return shape of 3
    def get_marginal(self, x):

        ckpt = tf.train.get_checkpoint_state(self.header+ os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        if len(np.array(x).shape)==2:
            marginal = self.sess.run(self.sm, feed_dict={self.x: [x]})
        else:
            marginal = self.sess.run(self.sm, feed_dict={self.x: x})
        return marginal

    # return shape of 1
    def get_confidence(self, x):
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        if isinstance(x,list):

            margs = self.sess.run(self.sm, feed_dict={self.x: x})
            margs = margs+np.finfo(float).eps
            margs = -np.sum(np.multiply(margs,np.log(margs)),axis=1)
            margs = np.minimum(1, margs)
            margs = np.maximum(0, margs)
            #conf  = np.mean(1-margs)
            conf = 1-margs
        else:

            margs = self.sess.run(self.sm, feed_dict={self.x: [x]})
            conf  = [1-np.maximum(0, np.minimum(1, - np.sum(margs * np.log(margs+np.finfo(float).eps))))]
        
        return conf
    
    def get_uncertainty(self, x, y):

        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        loss = self.sess.run(self.loss, feed_dict={self.x: [x], self.y: y})
        return loss

    # return shape of 10*64
    def get_xlogits(self, x, y):

        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        if len(np.array(x).shape)==2:
            logits = self.sess.run(self.xlogits, feed_dict={self.x: [x], self.y: [y]})
        else:
            logits = self.sess.run(self.xlogits, feed_dict={self.x: x, self.y: y})
        return logits


    def test(self, X_test, Y_true):

        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        acc = self.sess.run(self.accuracy, feed_dict={self.x: X_test, self.y: Y_true})
        # f_1 score and conf matrix
        return acc
    
    def get_f1_score(self, X_test, Y_true):
        ckpt = tf.train.get_checkpoint_state(self.header+ os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x: X_test}), 1)

        f1 = f1_score(Y_true, pred, average='macro')
        
        return f1

    def get_predictions_B(self, x, mode = 'B'):
    
        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep +  'test_{1}_{0}_/'.format(self.cvit, mode) + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one
    
        pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x: x}), 1)
        return pred

    def test_B(self, X_test, Y_true, mode = 'B'):


        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep + 'test_{1}_{0}_/'.format(self.cvit, mode) + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        acc = self.sess.run(self.accuracy, feed_dict={self.x: X_test, self.y: Y_true})
        # f_1 score and conf matrix
        return acc
    
    def get_f1_score_B(self, X_test, Y_true, mode = 'B'):
        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep + 'test_{1}_{0}_/'.format(self.cvit, mode)+self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x: X_test}), 1)

        f1 = f1_score(Y_true, pred, average='macro')
        
        return f1
    
    def get_confidence_B(self, x, mode = 'B'):
        
        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep + 'test_{1}_{0}_/'.format(self.cvit, mode)+ self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        if isinstance(x,list):

            margs = self.sess.run(self.sm, feed_dict={self.x: x})
            margs = margs+np.finfo(float).eps
            margs = -np.sum(np.multiply(margs,np.log(margs)),axis=1)
            margs = np.minimum(1, margs)
            margs = np.maximum(0, margs)
            conf = 1 - margs
            #conf  = np.mean(1-margs)

        else:

            margs = self.sess.run(self.sm, feed_dict={self.x: [x]})
            conf  = [1-np.maximum(0, np.minimum(1, - np.sum(margs * np.log(margs+np.finfo(float).eps))))]            

        return conf

    def get_uncertainty_B(self, x, y, mode = 'B'):

        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep + 'test_{1}_{0}_/'.format(self.cvit, mode)+ self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        loss = self.sess.run(self.loss, feed_dict={self.x: [x], self.y: y})
        return loss
