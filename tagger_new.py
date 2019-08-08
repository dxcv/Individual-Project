import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import slim
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from sklearn.metrics import f1_score

#os.environ['CUDA_VISIBLE_DEVICES']='0,1'  # Number of GPUs to run on
os.environ['CUDA_VISIBLE_DEVICES']='-1'

class Tagger(object):

    # take as input all the four features
    # [[257,10],[70,10],[27,10],[24,10]]
    # the way we compute the Q-value and updating agent keeps unchanged
    # four taggers concatenated in the one tagger, output (predictive marginals) from four taggers input to an fcl 
    # fcl output the true predictive marginal

    def __init__(self, model_file, n_steps, n_input, feature_number, training = True, epochs = 10, expnum = 0, cvit = ''):

        
        self.expnum = expnum
        self.header = 'EXP_{0}/new_checkpoint'.format(self.expnum)
        self.header_test = 'EXP_{0}/new_checkpoint_test'.format(self.expnum)
        self.model_file = model_file
        if not os.path.exists(self.header + os.sep + self.model_file):
            os.makedirs(self.header + os.sep + self.model_file)

        self.training = training


        #self.feature_shapes = [[257,10],[70,10],[27,10],[24,10]]
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

        self.n_input1 = 257
        self.n_input2 = 70
        self.n_input3 = 27
        self.n_input4 = 24
        self.n_steps = n_steps
        self.n_hidden = 64
        self.n_classes = 3


        # classifier
        def lstm(x, weights, biases, i):
            x = tf.unstack(x, self.n_steps, 1)
            with tf.variable_scope('tagger1_feature_{0}'.format(i)):
                lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
            with tf.variable_scope('tagger2_feature_{0}'.format(i)):
                self.outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                # take the self.outputs to the Q-network
            logitx = tf.stack(self.outputs)
            
            ##### average before fcl
            self.avg_outputs = tf.reduce_mean(tf.stack(self.outputs), 0)

            pred = tf.matmul(self.avg_outputs, weights['out']) + biases['out']
            
            #### average after fcl
            #self.matmul = []
            #for i in range(logitx.get_shape()[0]):
            #    self.matmul.append(tf.matmul(logitx[i], weights['out']) + biases['out'])

            
            # it is first averaged then input to the fcl
            #pred = tf.reduce_mean(tf.stack(self.matmul), 0)
            
            matmul = tf.zeros([10,3], tf.int32)
            #pred = tf.matmul(self.avg_outputs, weights['out']) + biases['out']

            return pred, logitx, matmul

        # individual lstm models
        self.x1 = tf.placeholder("float", [None, self.n_steps, self.n_input1])
        self.x2 = tf.placeholder("float", [None, self.n_steps, self.n_input2])
        self.x3 = tf.placeholder("float", [None, self.n_steps, self.n_input3])
        self.x4 = tf.placeholder("float", [None, self.n_steps, self.n_input4])

        # final output
        self.y = tf.placeholder("int32", [None])

    # Define weights
        with tf.variable_scope('weight_feature_1'):
            self.weights1 = {'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))}
        with tf.variable_scope('bias_feature_1'):
            self.biases1 = {'out': tf.Variable(tf.random_normal([self.n_classes]))}

        with tf.variable_scope('weight_feature_2'):
            self.weights2 = {'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))}
        with tf.variable_scope('bias_feature_2'):
            self.biases2 = {'out': tf.Variable(tf.random_normal([self.n_classes]))}

        with tf.variable_scope('weight_feature_3'):
            self.weights3 = {'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))}
        with tf.variable_scope('bias_feature_3'):
            self.biases3 = {'out': tf.Variable(tf.random_normal([self.n_classes]))}

        with tf.variable_scope('weight_feature_4'):
            self.weights4 = {'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))}
        with tf.variable_scope('bias_feature_4'):
            self.biases4 = {'out': tf.Variable(tf.random_normal([self.n_classes]))}

    # LSTM
        with tf.name_scope('lstm1'):
            self.pred1, self.xlogits1, self.xfcls1 = lstm(self.x1, self.weights1, self.biases1, 1)
        with tf.name_scope('lstm2'):
            self.pred2, self.xlogits2, self.xfcls2 = lstm(self.x2, self.weights2, self.biases2, 2)
        with tf.name_scope('lstm3'):
            self.pred3, self.xlogits3, self.xfcls3 = lstm(self.x3, self.weights3, self.biases3, 3)
        with tf.name_scope('lstm4'):
            self.pred4, self.xlogits4, self.xfcls4 = lstm(self.x4, self.weights4, self.biases4, 4)
        
        with tf.name_scope('sm1'):
            self.sm1 = tf.nn.softmax(self.pred1)
        with tf.name_scope('sm2'):
            self.sm2 = tf.nn.softmax(self.pred2)
        with tf.name_scope('sm3'):
            self.sm3 = tf.nn.softmax(self.pred3)
        with tf.name_scope('sm4'):
            self.sm4 = tf.nn.softmax(self.pred4)

        self.predss = tf.concat([tf.concat([self.pred1, self.pred2], axis=1),tf.concat([self.pred3, self.pred4], axis=1)],axis=1)

    # the sm1, sm2, sm3 and sm4 are prefictive marginals of the four lstm models
    # these are then inputed to a fcl


        self.confs = [self.sm1, self.sm2, self.sm3, self.sm4]
        a = tf.concat([self.sm1, self.sm2], axis=1)
        b = tf.concat([self.sm3, self.sm4], axis=1)

        self.net = tf.concat([a, b], axis=1)
        # TODO: only input in the self.net to the Q_network

        with tf.variable_scope('weight_feature_fcl'):
            self.weights = {'out': tf.Variable(tf.random_normal([12, self.n_classes]))}
        with tf.variable_scope('bias_feature_fcl'):
            self.biases = {'out': tf.Variable(tf.random_normal([self.n_classes]))}

        self.pred = tf.matmul(self.net, self.weights['out']) + self.biases['out']
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
        if data_x != []:
            data_x1 = []
            data_x2 = []
            data_x3 = []
            data_x4= []
            for i in range(len(data_x)):
                data_x1.append(data_x[i][0])
                data_x2.append(data_x[i][1])
                data_x3.append(data_x[i][2])
                data_x4.append(data_x[i][3])
            
        for i in range(self.epochs):
            step = 1
            while step * self.batch_size <= len(data_y):
                if data_x != []:
                    batch_x1 = data_x1[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x2 = data_x2[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x3 = data_x3[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x4 = data_x4[(step - 1) * self.batch_size:step * self.batch_size]
                else:
                    batch_x1 = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x2 = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x3 = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x4 = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                batch_y = data_y[(step - 1) * self.batch_size:step * self.batch_size]

                self.sess.run(self.optimizer, feed_dict={self.x1: batch_x1, self.x2: batch_x2, self.x3: batch_x3, self.x4: batch_x4, self.y: batch_y})

                if step % self.display_step == 0:
                    acc = self.sess.run(self.accuracy, feed_dict={self.x1: batch_x1, self.x2: batch_x2, self.x3: batch_x3, self.x4: batch_x4, self.y: batch_y})
                    loss = self.sess.run(self.loss, feed_dict={self.x1: batch_x1, self.x2: batch_x2, self.x3: batch_x3, self.x4: batch_x4, self.y: batch_y})
                    #print("Epoch: " + str(i + 1) + ", iter: " + str(
                    #    step * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                    #    loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                    self.loss_f.append(loss)
                    self.acc_f.append(100 * acc)

                step += 1

            if (i+1) % self.save_step == 0:
                self.saver.save(self.sess, self.header + os.sep + self.model_file + os.sep + 'model.ckpt', i+1)

    ### used during testing     
    def train_mode_B(self, data_x, data_y, feature_number):

        if not os.path.exists(self.header_test + os.sep + 'test_B_{0}_/'.format(self.cvit) +self.model_file):
            os.makedirs(self.header_test + os.sep + 'test_B_{0}_/'.format(self.cvit) +self.model_file)
            
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('error')
            '###############################ERROR#############################'

        if len(data_y) >= 150:
            self.batch_size = 32
        if data_x != []:
            data_x1 = []
            data_x2 = []
            data_x3 = []
            data_x4= []
            for i in range(len(data_x)):
                data_x1.append(data_x[i][0])
                data_x2.append(data_x[i][1])
                data_x3.append(data_x[i][2])
                data_x4.append(data_x[i][3])
            
        for i in range(self.epochs):
            step = 1
            while step * self.batch_size <= len(data_y):
                if data_x != []:
                    batch_x1 = data_x1[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x2 = data_x2[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x3 = data_x3[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x4 = data_x4[(step - 1) * self.batch_size:step * self.batch_size]
                else:
                    batch_x1 = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x2 = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x3 = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_x4 = data_x[(step - 1) * self.batch_size:step * self.batch_size]
                batch_y = data_y[(step - 1) * self.batch_size:step * self.batch_size]

                self.sess.run(self.optimizer, feed_dict={self.x1: batch_x1, self.x2: batch_x2, self.x3: batch_x3, self.x4: batch_x4, self.y: batch_y})

                if step % self.display_step == 0:
                    acc = self.sess.run(self.accuracy, feed_dict={self.x1: batch_x1, self.x2: batch_x2, self.x3: batch_x3, self.x4: batch_x4, self.y: batch_y})
                    loss = self.sess.run(self.loss, feed_dict={self.x1: batch_x1, self.x2: batch_x2, self.x3: batch_x3, self.x4: batch_x4, self.y: batch_y})
                    #print("Epoch: " + str(i + 1) + ", iter: " + str(
                    #    step * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                    #    loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                    self.loss_f.append(loss)
                    self.acc_f.append(100 * acc)

                step += 1

            if (i+1) % self.save_step == 0:
                self.saver.save(self.sess, self.header_test + os.sep  + 'test_B_{0}_/'.format(self.cvit) + self.model_file + os.sep + 'model.ckpt', i+1)

    def get_predictions(self, x):
        
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one
        if len(np.array(x[0]).shape)==2:
            pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x1: [x[0]], self.x2: [x[1]],self.x3: [x[2]],self.x4: [x[3]]}), 1)
        else:
            pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x1: x[0], self.x2: x[1],self.x3: x[2],self.x4: x[3]}), 1)
        return pred
    
    def get_marginal(self, x):

        ckpt = tf.train.get_checkpoint_state(self.header+ os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        if len(np.array(x[0]).shape)==2:
            # should we use net or marginals?
            marginal = self.sess.run(self.net, feed_dict={self.x1: [x[0]], self.x2: [x[1]],self.x3: [x[2]],self.x4: [x[3]]})
        else:
            #x = np.array(x)
            marginal = self.sess.run(self.net, feed_dict={self.x1: x[0], self.x2: x[1],self.x3: x[2],self.x4: x[3]})
        return marginal

    def get_confidence(self, x):
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        if len(np.array(x[0]).shape)==3:
            margs_all = self.sess.run(self.confs, feed_dict={self.x1: x[0], self.x2: x[1], self.x3: x[2], self.x4: x[3]})
            confs = []
            for margs in margs_all:
                margs = margs+np.finfo(float).eps
                margs = -np.sum(np.multiply(margs,np.log(margs)),axis=1)
                margs = np.minimum(1, margs)
                margs = np.maximum(0, margs)
                #conf  = np.mean(1-margs)
                conf = 1-margs
                confs.append(conf.reshape(-1,1))
            confs = np.concatenate(confs, axis = 1)
        else:
            margs_all = self.sess.run(self.confs, feed_dict={self.x1: [x[0]], self.x2: [x[1]],self.x3: [x[2]],self.x4: [x[3]]})
            confs = []
            for margs in margs_all:
                conf  = [1-np.maximum(0, np.minimum(1, - np.sum(margs * np.log(margs+np.finfo(float).eps))))]
                confs = conf + confs     
        return confs
    
    def get_uncertainty(self, x, y):

        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        loss = self.sess.run(self.loss, feed_dict={self.x1: [x[0]], self.x2: [x[1]],self.x3: [x[2]],self.x4: [x[3]], self.y: y})
        return loss

    def get_xlogits(self, x, y):

        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        if len(np.array(x[0]).shape)==2:    
            logits = self.sess.run(self.xlogits1, feed_dict={self.x1: [x[0]], self.y: [y]})
        else:
            logits = self.sess.run(self.xlogits1, feed_dict={self.x1: x[0], self.y: y})
        return logits

    def get_xfcls(self, x):
        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        if len(np.array(x[0]).shape)==2:
            xfcls = self.sess.run(self.xfcls1, feed_dict={self.x1: [x[0]]})
        else:
            xfcls = self.sess.run(self.xfcls1, feed_dict={self.x1: x[0]})
        return xfcls


    def test(self, X_test, Y_true):

        ckpt = tf.train.get_checkpoint_state(self.header + os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        if len(np.array(X_test[0]).shape) == 2:
            acc = self.sess.run(self.accuracy, feed_dict={self.x1: [X_test[0]], self.x2: [X_test[1]], self.x3: [X_test[2]], self.x4: [X_test[3]], self.y: [Y_true]})
        else:
            acc = self.sess.run(self.accuracy, feed_dict={self.x1: X_test[0], self.x2: X_test[1], self.x3: X_test[2], self.x4: X_test[3], self.y: Y_true})
           
        # f_1 score and conf matrix
        return acc
    
    def get_f1_score(self, X_test, Y_true):
        ckpt = tf.train.get_checkpoint_state(self.header+ os.sep + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one
        if len(np.array(X_test[0]).shape) == 2:
            pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x1: [X_test[0]], self.x2: [X_test[1]], self.x3: [X_test[2]], self.x4: [X_test[3]]}), 1)
        else:
            pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x1: X_test[0], self.x2: X_test[1], self.x3: X_test[2], self.x4: X_test[3]}), 1)
            

        f1 = f1_score(Y_true, pred, average='macro')
        
        return f1

    def get_predictions_B(self, x):
    
        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep +  'test_B_{0}_/'.format(self.cvit) + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one
    
        if len(np.array(x[0]).shape)==2:
            pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x1: [x[0]], self.x2: [x[1]],self.x3: [x[2]],self.x4: [x[3]]}), 1)
        else:
            pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x1: x[0], self.x2: x[1],self.x3: x[2],self.x4: x[3]}), 1)
        return pred

    def test_B(self, X_test, Y_true):


        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep + 'test_B_{0}_/'.format(self.cvit) + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        if len(np.array(X_test[0]).shape) == 2:
            acc = self.sess.run(self.accuracy, feed_dict={self.x1: [X_test[0]], self.x2: [X_test[1]], self.x3: [X_test[2]], self.x4: [X_test[3]], self.y: [Y_true]})
        else:
            acc = self.sess.run(self.accuracy, feed_dict={self.x1: X_test[0], self.x2: X_test[1], self.x3: X_test[2], self.x4: X_test[3], self.y: Y_true})
           
        # f_1 score and conf matrix
        return acc
    
    def get_f1_score_B(self, X_test, Y_true):
        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep + 'test_B_{0}_/'.format(self.cvit) + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # no need to restore since we use the last one

        if len(np.array(X_test[0]).shape) == 2:
            pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x1: [X_test[0]], self.x2: [X_test[1]], self.x3: [X_test[2]], self.x4: [X_test[3]]}), 1)
        else:
            pred = np.argmax(self.sess.run(self.pred, feed_dict={self.x1: X_test[0], self.x2: X_test[1], self.x3: X_test[2], self.x4: X_test[3]}), 1)
            

        f1 = f1_score(Y_true, pred, average='macro')
        
        return f1
    

    def get_marginal_B(self, x):
        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep + 'test_B_{0}_/'.format(self.cvit) + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        if len(np.array(x[0]).shape)==2:
            # should we use net or marginals?
            marginal = self.sess.run(self.net, feed_dict={self.x1: [x[0]], self.x2: [x[1]],self.x3: [x[2]],self.x4: [x[3]]})
        else:
            #x = np.array(x)
            marginal = self.sess.run(self.net, feed_dict={self.x1: x[0], self.x2: x[1],self.x3: x[2],self.x4: x[3]})
        return marginal

    def get_confidence_B(self, x):
        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep + 'test_B_{0}_/'.format(self.cvit) + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        if len(np.array(x[0]).shape)==3:
            margs_all = self.sess.run(self.confs, feed_dict={self.x1: x[0], self.x2: x[1], self.x3: x[2], self.x4: x[3]})
            confs = []
            for margs in margs_all:
                margs = margs+np.finfo(float).eps
                margs = -np.sum(np.multiply(margs,np.log(margs)),axis=1)
                margs = np.minimum(1, margs)
                margs = np.maximum(0, margs)
                #conf  = np.mean(1-margs)
                conf = 1-margs
                confs.append(conf.reshape(-1,1))
            confs = np.concatenate(confs, axis = 1)
        else:
            margs_all = self.sess.run(self.confs, feed_dict={self.x1: [x[0]], self.x2: [x[1]],self.x3: [x[2]],self.x4: [x[3]]})
            confs = []
            for margs in margs_all:
                conf  = [1-np.maximum(0, np.minimum(1, - np.sum(margs * np.log(margs+np.finfo(float).eps))))]
                confs = conf + confs     
        return confs

    def get_uncertainty_B(self, x, y):

        ckpt = tf.train.get_checkpoint_state(self.header_test + os.sep + 'test_B_{0}_/'.format(self.cvit) + self.model_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        loss = self.sess.run(self.loss, feed_dict={self.x1: [x[0]], self.x2: [x[1]],self.x3: [x[2]],self.x4: [x[3]], self.y: y})
        return loss
