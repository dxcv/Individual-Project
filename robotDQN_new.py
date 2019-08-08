# auto-maj voting strategy, the input to the q network is the predictive marginals and confidence
# input shape is (16): (12+4)

import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.contrib import rnn
import os
from sklearn.preprocessing import PolynomialFeatures
#os.environ['CUDA_VISIBLE_DEVICES']='0,1'  # Number of GPUs to run on
os.environ['CUDA_VISIBLE_DEVICES']='-1'

# Hyper Parameters:
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 32.  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 300  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 1
UPDATE_FREQ = 10

# or alternative:
# FINAL_EPSILON = 0.0001  # final value of epsilon
# INITIAL_EPSILON = 0.01  # starting value of epsilon
# UPDATE_TIME = 100
EXPLORE = 10000.  # frames over which to anneal epsilon



class RobotLSTMQ:

    def __init__(self, actions, features, content, poly, logit, fcls, ntype, expnum):
        #print("Creating a robot: LSTM-Q")
        # replay memory
        self.replay_memory = deque()
        self.time_step = 0.
        self.action = actions
        self.ntype = ntype #'d1qn' # ['d1qn', 'd2qn', 'd3qn']
        self.expnum = expnum

        self.train_step = UPDATE_FREQ # UPDATE FREQUENCY FOR THE Q NETWORK
        self.content = content
        self.logit = logit
        self.poly = poly
        self.fcls = fcls
        #number of features selected
        self.n_features = len(features)
        #shape of all features
        self.features = features
        self.feature_shape = [[378,10],[257,10],[70,10],[27,10],[24,10]]
        self.epsilon = INITIAL_EPSILON
        self.observe = OBSERVE
        self.expand = [15,45,91,153]
        self.num_classes = 3
        self.n_hidden  = 32
        self.batch_size = 32
        ### create two q networks instead        
        self.mainQN = q_network(self, scope = 'mainQN')

        if self.ntype == 'd2qn' or self.ntype == 'd3qn': #d1qn, d3qn
            self.targetQN = q_network(self, scope ='targetQN') 
            self.trainables = tf.trainable_variables()
            self.targetOps  = self.updateTargetGraph(tau=0.01)

        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        # ? multiple graphs: how to initialise variables 
        self.sess.run(tf.global_variables_initializer())


        ################################################
    def updateTargetGraph(self, tau = 0.001):        
        tfVars = self.trainables        
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars//2]):
            op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
        return op_holder

    def updateTarget(self):
        for op in self.targetOps:
            self.sess.run(op)

    def update(self, observation, action, reward, observation2, terminal):
        self.current_state = observation
        new_state = observation2
        self.replay_memory.append((self.current_state, action, reward, new_state, terminal))
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()

        if self.time_step > self.observe and self.time_step % self.train_step == 0.0:
            # Train the network            
            self.train_qnetwork()
            #print('DQN-trained')

        self.current_state = new_state
        self.time_step += 1


    def train_qnetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        next_state_sent_batch = []
        next_state_confidence_batch = []
        next_state_predictions_batch = []
        next_state_logits_batch = []
        next_state_fcls_batch = []

        for item in next_state_batch:
            sent, confidence, predictions, logits, fcls = item
            next_state_sent_batch.append(sent)
            # next state_sent_batch has size 
            next_state_confidence_batch.append(confidence)
            next_state_predictions_batch.append(predictions)
            next_state_logits_batch.append(logits)
            next_state_fcls_batch.append(fcls)


        # Step 2: calculate y
        #next_state_predictions_batch = np.reshape(next_state_predictions_batch, (len(next_state_sent_batch),self.n_features,3))


        for i in range(self.batch_size):
            next_state_sent_batch[i] = np.concatenate((next_state_sent_batch[i]),axis=1)
        concatenate_x = np.array(next_state_sent_batch)
        
        #concatenate_x = np.array(next_state_sent_batch)[:,-1]
        concatenate_predictions = np.squeeze(next_state_predictions_batch, axis=1)


        y_batch = []
        next_action_batch_main = []
        qvalue_batch_target = []


        if self.content:
            # TODO: 
            # if content, remove the state_confidence, predictions
            # computing the qvalue for the next state! Instead, we should compute the action by the main network 
            # and use it to select the q-value by the target network
            
            #qvalue_batch = self.sess.run(self.mainQnetwork.qvalue, feed_dict={self.mainQnetwork.x: concatenate_x, 
            #    self.mainQnetwork.state_confidence: next_state_confidence_batch, self.mainQnetwork.predictions: list(concatenate_predictions)})

            if self.ntype == 'd1qn' or self.ntype == 'd4qn':
                #qvalue_batch_target = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x: concatenate_x, 
                #    self.mainQN.state_confidence: next_state_confidence_batch, self.mainQN.predictions: list(concatenate_predictions)})
                if self.logit:
                    qvalue_batch_target = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x: concatenate_x, self.mainQN.xlogits: next_state_logits_batch})
                else:
                    qvalue_batch_target = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x: concatenate_x})

            elif self.ntype == 'd2qn' or self.ntype == 'd3qn':
                next_action_batch_main = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.x: concatenate_x, 
                    self.mainQN.state_confidence: next_state_confidence_batch, self.mainQN.predictions: list(concatenate_predictions)})
                qvalue_batch_target = self.sess.run(self.targetQN.qvalue, feed_dict={self.targetQN.x: concatenate_x, 
                    self.targetQN.state_confidence: next_state_confidence_batch, self.targetQN.predictions: list(concatenate_predictions)})
            else:                
                print("** Q-learning method not defined.")
                raise SystemExit

        elif self.logit:
            # shape of 10*64
            # input into the lstm
            # there are four, so concatenate to 10*(64*4)
            if self.ntype == 'd1qn' or self.ntype == 'd4qn':
                qvalue_batch_target = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xlogits: next_state_logits_batch})
            elif self.ntype == 'd2qn' or self.ntype == 'd3qn':
                next_action_batch_main = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.x: concatenate_x, 
                    self.mainQN.state_confidence: next_state_confidence_batch, self.mainQN.predictions: list(concatenate_predictions)})
                qvalue_batch_target = self.sess.run(self.targetQN.qvalue, feed_dict={self.targetQN.x: concatenate_x, 
                    self.targetQN.state_confidence: next_state_confidence_batch, self.targetQN.predictions: list(concatenate_predictions)})
            else:                
                print("** Q-learning method not defined.")
                raise SystemExit 
        elif self.fcls:
            if self.ntype == 'd1qn':
                qvalue_batch_target = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xfcls: next_state_fcls_batch})
            else:                
                print("** Q-learning method not defined.")
                raise SystemExit
        else:
            if self.poly:
                enc_total_in = np.concatenate((next_state_confidence_batch, list(concatenate_predictions)), axis=1)
                poly = PolynomialFeatures(2)
                expanded_enc_total_in = poly.fit_transform(enc_total_in)
                #print(expanded_enc_total_in.shape)
                if self.ntype == 'd1qn' or self.ntype == 'd4qn':
                    qvalue_batch_target = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.enc_total: expanded_enc_total_in})
                elif self.ntype == 'd2qn' or self.ntype == 'd3qn':
                    next_action_batch_main = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.enc_total: expanded_enc_total_in})
                    qvalue_batch_target = self.sess.run(self.targetQN.qvalue, feed_dict={self.targetQN.enc_total: expanded_enc_total_in})
                else:
                    print("** Q-learning method not defined.")
                    raise SystemExit


            else:
                if self.ntype == 'd1qn' or self.ntype == 'd4qn':
                    #qvalue_batch_target = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.state_confidence: np.squeeze(next_state_confidence_batch),
                    #    self.mainQN.predictions: list(np.squeeze(concatenate_predictions))})
                    qvalue_batch_target = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.state_confidence: np.squeeze(next_state_confidence_batch), self.mainQN.predictions: list(np.squeeze(concatenate_predictions))})
                elif self.ntype == 'd2qn' or self.ntype == 'd3qn':
                    next_action_batch_main = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.state_confidence: next_state_confidence_batch,
                        self.mainQN.predictions: list(concatenate_predictions)})
                    qvalue_batch_target = self.sess.run(self.targetQN.qvalue, feed_dict={self.targetQN.state_confidence: next_state_confidence_batch,
                        self.targetQN.predictions: list(concatenate_predictions)})



        # HERE WE NEED TO INCLUDE THE TARGET NETWORK!!

        if self.ntype == 'd1qn' or self.ntype == 'd4qn':
            doubleQ = np.argmax(qvalue_batch_target,1)
        elif self.ntype == 'd2qn' or self.ntype == 'd3qn':
            doubleQ = qvalue_batch_target[range(self.batch_size),next_action_batch_main]                
        else:                
            raise NameError('No q-learning defined') 
                    

         

        # if terminal then we multiply with 0 the qvalue part

        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * doubleQ[i])
                
        # inspect errors
        

        sent_batch = []
        confidence_batch = []
        predictions_batch = []
        logits_batch = []
        fcls_batch = []
        for item in state_batch:
            sent, confidence, predictions, logits, fcls = item
            sent_batch.append(sent)
            confidence_batch.append(confidence)
            predictions_batch.append(predictions)
            logits_batch.append(logits)
            fcls_batch.append(fcls)
        
        #concatenate_x = np.concatenate((next_state_sent_batch[0]),axis=1)

        for i in range(self.batch_size):
            sent_batch[i] = np.concatenate((sent_batch[i]),axis=1)
        concatenate = np.array(sent_batch)
        
        concatenate_predictions = np.squeeze(predictions_batch, axis=1)

        if self.content:

            # update the main network
            #self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch, 
            #    self.mainQN.action_input: action_batch, self.mainQN.x: concatenate,
            #    self.mainQN.state_confidence: confidence_batch, self.mainQN.predictions:list(concatenate_predictions)})
            if self.logit:
                self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch, 
                self.mainQN.action_input: action_batch, self.mainQN.x: concatenate, self.mainQN.xlogits: logits_batch})
            else:
                self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch, 
                self.mainQN.action_input: action_batch, self.mainQN.x: concatenate})
        elif self.logit:
            self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch, 
                self.mainQN.action_input: action_batch, self.mainQN.xlogits: logits_batch})
        elif self.fcls:
            self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch, 
                self.mainQN.action_input: action_batch, self.mainQN.xfcls: fcls_batch})
        else:
        
            if self.poly:
                enc_total_in = np.concatenate((confidence_batch, list(concatenate_predictions)), axis=1)
                poly = PolynomialFeatures(2)
                expanded_enc_total_in = poly.fit_transform(enc_total_in)
                #print(expanded_enc_total_in.shape)
                self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch, self.mainQN.action_input: action_batch,
                    self.mainQN.enc_total: expanded_enc_total_in})
            else:
                self.sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.q_next: y_batch, self.mainQN.action_input: action_batch,
                    self.mainQN.state_confidence: confidence_batch, self.mainQN.predictions:list(concatenate_predictions)})

        if self.ntype == 'd2qn' or self.ntype == 'd3qn':
            self.updateTarget()
        

        ## save network every 10000 iteration
        ## if self.time_step % 10000 == 0:
        ##    self.saver.save(self.sess, './' +
        ##                    'network' + '-dqn', global_step=self.time_step)



    def get_action(self, observation):
        #print("LSTM-DQN is smart.")
        self.current_state = observation
        sent, confidence, predictions, logits, fcls = self.current_state
        # print sent, confidence, predictions
        if self.content:
            concatenate = sent[-1]
            for i in range(self.n_features-1):
                concatenate = np.concatenate((concatenate, sent[i]),axis=1)
            #qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x:[concatenate],
            #    self.mainQN.state_confidence: [confidence], self.mainQN.predictions: predictions})[0]
            if self.logit:
                qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x:[concatenate],self.mainQN.xlogits:[logits]})[0]
            else:
                qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x:[concatenate]})
        elif self.logit:
            qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xlogits:[logits]})

        elif self.fcls:
            qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xfcls:[fcls]})
        else:
            if self.poly:
                #print(np.array([confidence]).shape)
                #print(np.array(predictions).shape)
                enc_total_in = np.concatenate(([confidence], predictions), axis=1)
                poly = PolynomialFeatures(2)
                expanded_enc_total_in = poly.fit_transform(enc_total_in)
                qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.enc_total: expanded_enc_total_in})
            else:
                #qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.state_confidence: [confidence], self.mainQN.predictions: predictions})[0]
                qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.state_confidence: [confidence], self.mainQN.predictions: predictions})[0]

        
        action = np.zeros(self.action)
        action_index = 0
        # if self.timeStep % FRAME_PER_ACTION == 0:
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.action)
            action[action_index] = 1
        else:
            action_index = np.argmax(qvalue)
            action[action_index] = 1
        # else:
        #    action[0] = 1 # do nothing

        # change epsilon
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action


    # we should save only the main Q network --- as during inference we don't use the target network 
    def save_Q_network(self, model_val):
        npath = 'EXP_{0}/new_checkpoint'.format(self.expnum)
        if not os.path.exists(npath + os.sep + "Q_" + model_val):
            os.makedirs(npath + os.sep + "Q_" + model_val)
        self.saver.save(self.sess, npath + os.sep + "Q_" + model_val + os.sep + 'model.ckpt')
        #print('save the q network')
        
    def test_get_action(self, model_val, observation):
        npath = 'EXP_{0}/new_checkpoint'.format(self.expnum)
        ckpt = tf.train.get_checkpoint_state(npath + os.sep + "Q_" + model_val)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        #print('load the q network')
        #print("LSTM-DQN is smart.")
        self.current_state = observation
        sent, confidence, predictions, logits, fcls = self.current_state
        # print sent, confidence, predictions
        if self.content:
            concatenate = sent[-1]
            for i in range(self.n_features-1):
                concatenate = np.concatenate((concatenate, sent[i]),axis=1)
            #qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x:[concatenate],
            #    self.mainQN.state_confidence: [confidence], self.mainQN.predictions: predictions})[0]
            if self.logit:
                qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x:[concatenate],self.mainQN.xlogits:[logits]})[0]
            else:
                qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.x:[concatenate]})
        elif self.logit:
            qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xlogits:[logits]})

        elif self.fcls:
            qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.xfcls:[fcls]})
        else:
            if self.poly:
                #print(np.array([confidence]).shape)
                #print(np.array(predictions).shape)
                enc_total_in = np.concatenate(([confidence], predictions), axis=1)
                poly = PolynomialFeatures(2)
                expanded_enc_total_in = poly.fit_transform(enc_total_in)
                qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.enc_total: expanded_enc_total_in})
            else:
                #qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.state_confidence: [confidence], self.mainQN.predictions: predictions})[0]
                qvalue = self.sess.run(self.mainQN.qvalue, feed_dict={self.mainQN.state_confidence: [confidence], self.mainQN.predictions: predictions})[0]

        
        action = np.zeros(self.action)
        action_index = 0
        # if self.timeStep % FRAME_PER_ACTION == 0:
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.action)
            action[action_index] = 1
        else:
            action_index = np.argmax(qvalue)
            action[action_index] = 1
        # else:
        #    action[0] = 1 # do nothing

        # change epsilon
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action


class q_network():

    # read an input

    def __init__(self, robot, scope):

        self.num_classes = robot.num_classes
        self.n_hidden  = robot.n_hidden
        self.n_features = robot.n_features
        self.content = robot.content
        self.features = robot.features
        self.feature_shape = robot.feature_shape
        self.action = robot.action
        self.poly = robot.poly
        self.fcls = robot.fcls
        self.expand = robot.expand
        self.ntype = robot.ntype
        self.logit = robot.logit

        ### added this to have consistent inits 
        random.seed(0)

        if self.content:
            def lstm1(x, weights, biases):
                x = tf.unstack(x,10, 1)
                with tf.variable_scope(scope+'_lstmc1'):
                    lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
                with tf.variable_scope(scope+'_lstmc2'):
                    self.outputs1, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                self.avg_outputs1 = tf.reduce_mean(tf.stack(self.outputs1), 0)
                
                pred = tf.matmul(self.avg_outputs1, weights['out']) + biases['out']

                return pred

            inputs_all = 0
            for i in self.features:
                inputs_all = self.feature_shape[i][0] + inputs_all
            
            #print('x shape:', self.feature_shape[0][1], inputs_all)

            self.x = tf.placeholder(
                    tf.float32, [None, self.feature_shape[i][1], inputs_all], name="input_x")

            # self.state_confidence = tf.placeholder(
            #         tf.float32, [None, self.n_features], name="input_confidence")
            # self.predictions = tf.placeholder(
            #         tf.float32, [None, self.num_classes*self.n_features], name="input_predictions")
            
            #have n_features placeholders            
            
            
            # network weights
            # size of a input = 10*3
            self.state_len = inputs_all
            self.weight_proj_c = {'out': tf.Variable(tf.random_normal([self.n_hidden, self.state_len]))}
            self.biases_proj_c = {'out': tf.Variable(tf.random_normal([self.state_len]))}

            # if only content, input to lstm is 10*378
            self.enc_s = lstm1(self.x, self.weight_proj_c, self.biases_proj_c)
            print('shape enc_s', self.enc_s.get_shape())

            if self.logit:
                def lstm2(x, weights, biases):
                    x = tf.unstack(x, 10, 1)
                    with tf.variable_scope(scope+'_lstml1'):
                        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
                    with tf.variable_scope(scope+'_lstml2'):
                        self.outputs2, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                    self.avg_outputs2 = tf.reduce_mean(tf.stack(self.outputs2), 0)
                    
                    pred = tf.matmul(self.avg_outputs2, weights['out']) + biases['out']
                    
                    return pred

                self.xlogits = tf.placeholder(tf.float32, [None, 10, self.n_features*64], name="x_logits")
                self.weight_proj_l = {'out': tf.Variable(tf.random_normal([self.n_hidden, 64*self.n_features]))}
                self.biases_proj_l = {'out': tf.Variable(tf.random_normal([64*self.n_features]))}
            # confidence is a scalar
            #self.enc_total = tf.concat((self.enc_s, self.state_confidence, self.predictions), axis=1)

                self.final_logits = lstm2(self.xlogits, self.weight_proj_l, self.biases_proj_l,i) #should have shape of 32

                # if content and logit, shape input to lstm should be 10*(378+256)?
                self.enc_total = tf.concat([self.final_logits, self.enc_s],1)
                #self.enc_total = tf.reshape(self.enc_total, [-1,64*self.n_features])
                print('shape enc_total', self.enc_total.get_shape())
            else:
                self.enc_total = self.enc_s

        # if no content, only logits
        elif self.logit:
            def lstm2(x, weights, biases):
                x = tf.unstack(x, 10, 1)
                with tf.variable_scope(scope+'_lstml1'):
                    lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
                with tf.variable_scope(scope+'_lstml2'):
                    self.outputs2, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                self.avg_outputs2 = tf.reduce_mean(tf.stack(self.outputs2), 0)
                
                pred = tf.matmul(self.avg_outputs2, weights['out']) + biases['out']
                
                return pred

            self.xlogits = tf.placeholder(tf.float32, [None, 10, self.n_features*64], name="x_logits")
            self.weight_proj_l = {'out': tf.Variable(tf.random_normal([self.n_hidden, 64*self.n_features]))}
            self.biases_proj_l = {'out': tf.Variable(tf.random_normal([64*self.n_features]))}
            # confidence is a scalar
            #self.enc_total = tf.concat((self.enc_s, self.state_confidence, self.predictions), axis=1)

            self.final_logits = lstm2(self.xlogits, self.weight_proj_l, self.biases_proj_l) #should have shape of 32
            self.enc_total = self.final_logits
            # only logits, should have a shape of 64*4
            print('shape_enc_total', self.enc_total.get_shape())

        elif self.fcls:
            def lstm2(x, weights, biases):
                x = tf.unstack(x, 10, 1)
                with tf.variable_scope(scope+'_lstmf1'):
                    lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
                with tf.variable_scope(scope+'_lstmf2'):
                    self.outputs3, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                self.avg_outputs3 = tf.reduce_mean(tf.stack(self.outputs3), 0)
                
                pred = tf.matmul(self.avg_outputs3, weights['out']) + biases['out']
                
                return pred

            self.xfcls = tf.placeholder(tf.float32, [None, 10, self.n_features*3], name="x_logits")
            self.weight_proj_f = {'out': tf.Variable(tf.random_normal([self.n_hidden, 3*self.n_features]))}
            self.biases_proj_f = {'out': tf.Variable(tf.random_normal([3*self.n_features]))}
            # confidence is a scalar
            #self.enc_total = tf.concat((self.enc_s, self.state_confidence, self.predictions), axis=1)

            self.final_xfcls = lstm2(self.xfcls, self.weight_proj_f, self.biases_proj_f) #should have shape of 32
            self.enc_total = self.final_xfcls
            # only logits, should have a shape of 64*4
            print('shape_enc_total', self.enc_total.get_shape())

        else:
            if self.poly:
                self.enc_total = tf.placeholder(
                    tf.float32, [None, self.expand[self.n_features-1]], name="input_expanded")
            else:
                self.state_confidence = tf.placeholder(
                    tf.float32, [None, 4], name="input_confidence")
                self.predictions = tf.placeholder(
                    tf.float32, [None, 4*self.num_classes], name="input_predictions")
                self.enc_total = self.predictions
                self.enc_total = tf.concat((self.state_confidence, self.predictions), axis=1)


        with tf.variable_scope(scope+'_robot'):
            self.w_fc2 = self.weight_variable([self.enc_total.get_shape().as_list()[-1], self.action])
            self.b_fc2 = self.bias_variable([self.action])
            if self.ntype == 'd3qn' or self.ntype == 'd4qn':
                self.w_fc3 = self.weight_variable([self.enc_total.get_shape().as_list()[-1], 1])


        
        if self.ntype == 'd3qn' or self.ntype == 'd4qn':
            # Extension to  include Dueling DDQN (Advantage can be negative, that is fine. It means that the action a in A(s,a) is a worse choice than the current policy's)
            # we start with the last layer -- enc_total

            # value function
            self.avalue  = tf.matmul(self.enc_total, self.w_fc2) + self.b_fc2

            # advantage 
            self.vvalue  = tf.matmul(self.enc_total, self.w_fc3) #+ self.b_fc3        # this bias may cause instability
            #Then combine them together to get our final Q-values.
            self.qvalue = self.vvalue + tf.subtract(self.avalue,tf.reduce_mean(self.avalue,axis=1,keep_dims=True))
            #self.qvalue = tf.subtract(self.avalue,tf.reduce_mean(self.avalue,axis=1,keep_dims=True))


        else:
            # Q Value layer
            self.qvalue  = tf.matmul(self.enc_total, self.w_fc2) + self.b_fc2

        #self.qvalue  = tf.matmul(self.enc_total, self.w_fc2) + self.b_fc2

        self.predict = tf.argmax(self.qvalue,1)
        #self.q_dist  = tf.nn.softmax(self.qvalue)

        # action input
        self.action_input = tf.placeholder("float", [None, self.action])
        self.q_action = tf.reduce_sum(tf.multiply(self.qvalue, self.action_input), reduction_indices=1)

        # reward input
        self.q_next = tf.placeholder("float", [None])              

        # error function

        loss = tf.reduce_sum(tf.square(self.q_next - self.q_action))
        #loss = tf.losses.huber_loss(self.q_next, self.q_action) ### check the error magnitudes + update model function

        # train method
        with tf.variable_scope('adam2'):
            trainer = tf.train.AdamOptimizer(1e-3)
            #trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)

        self.updateModel = trainer.minimize(loss)
        
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)