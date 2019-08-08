#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:44:55 2019

@author: beryl
"""
###
#added the cycling
###
import numpy as np
import sys
import random
from tagger import Tagger
import tensorflow as tf
import helper
import csv
import os
from collections import Counter
from sklearn.metrics import f1_score
#os.environ['CUDA_VISIBLE_DEVICES']='0,1'  # Number of GPUs to run on
os.environ['CUDA_VISIBLE_DEVICES']='-1'




class Env:

    def __init__(self, story, test, dev, budget, model_ver, models, feature_number, accum, expnum, cvit, method):
        ###load data
        self.train_x_all, self.train_y_all = story
        self.test_x_all, self.test_y_all = test
        self.dev_x_all, self.dev_y_all = dev

        self.method = method
        self.accum = accum #accumlate or not

        self.order = np.linspace(0, len(self.train_x_all[0])-1, len(self.train_x_all[0]))
        self.order = self.order.astype(int)
        self.cvit = cvit
        random.seed(self.cvit)
        random.shuffle(self.order)
        # if re-order, use random.shuffle(self.order)

        ### budget of query set
        self.budget = budget
        self.queried_times = 0

        ### query set
        self.queried_set_x = []
        self.queried_set_y = []
        self.query_set_num = 0

        ### use to control when to train the classifiers
        self.train_step = budget

        ### count the # of action 0 and # of action 1
        self.count_action0 = 0
        self.count_action1 = 0
        
        ### the features (modalities)
        self.feature = feature_number
        

        ### Initialise to episode 0
        self.episode = 0
        ### current frame
        self.current_frame = 0
        ### terminating condition, if this episode is ending, terminal=True
        self.terminal = False
        ### flag about making query or not
        self.make_query = False
        ### number of round, maximum 5
        self.rounds = 0
        ### accuracy of taggers
        self.performance = []
        ### accumulate the rewards for each action to calculate the total reward for each episode
        self.rAll = []

        ### use to store the cross counts 
        self.cross_counts_correct_train = np.zeros(shape=(len(self.feature)+1,len(self.feature)+2)).astype(int)
        self.cross_counts_incorrect_train = np.zeros(shape=(len(self.feature)+1,len(self.feature)+2)).astype(int)
        self.cross_counts_correct_test = np.zeros(shape=(len(self.feature)+1,len(self.feature)+2)).astype(int)
        self.cross_counts_incorrect_test = np.zeros(shape=(len(self.feature)+1,len(self.feature)+2)).astype(int)


        ### naming the csv files
        #path = 'C_Q_{0}_content/test_{1}/'.format('d1qn', cvit)
        path = 'EXP_{0}/records/'.format(expnum)
        if not os.path.exists(path):
            os.makedirs(path)
        self.csvfile = path+str(model_ver)+'.csv'
        self.csv_correct_train = path+'count_correct_train'+str(model_ver)+'.csv'
        self.csv_incorrect_train = path+'count_incorrect_train'+str(model_ver)+'.csv'
        self.csv_correct_test = path+'count_correct_test'+str(model_ver)+'.csv'
        self.csv_incorrect_test = path+'count_incorrect_test'+str(model_ver)+'.csv'

    ### function that returns the current observation (state), [content, confidence, marginal predictives]
    def get_frame(self, model):
        self.make_query = False
        frame = []
        confidence_all = []
        predictions_all = []
        logits = []
        for i in range(len(model)):
            frame.append(self.train_x_all[i][self.order[self.current_frame]])
            confidence_all.append(model[i].get_confidence([frame[i]]))
            predictions = model[i].get_marginal(frame[i])
            logits.append(model[i].get_xlogits(frame[i], self.train_y_all[self.order[self.current_frame]])) #shape of 10 * 64, after all, logits has a shape of 4*10*64
            for j in range(3):
                predictions_all.append(predictions[0][j])
        predictions_all = np.array(predictions_all).reshape(1,3*len(model))
        confidence_all = np.array(confidence_all).reshape(len(model),)
        if len(self.feature) > 1:
            obervation = [frame, confidence_all, predictions_all, np.squeeze(logits)]
        else:
            obervation = [frame, confidence_all, predictions_all, [np.squeeze(logits)]]
        return obervation 
    
    ### function to compute the reward, return reward, next_observation, terminal flag
    def feedback(self, action, model):
        reward = 0.
        self.terminal = False
        
        if action[1] == 1:
            self.count_action1 += 1
            self.make_query =True
            self.query()
            #self.performance,_ = self.get_performance(model)
            reward = -0.05
        else:
            #self.performance,_ = self.get_performance(model)
            self.count_action0 += 1

            label = self.train_y_all[self.order[self.current_frame]]
            pred = []
            confidences = []
            for i in range(len(model)):
                pred.append(model[i].get_predictions([self.train_x_all[i][self.order[self.current_frame]]])[0])
                confidences.append(model[i].get_confidence(self.train_x_all[i][self.order[self.current_frame]])[0])

            multi_pred = helper.voting_multi_instance(self, pred, confidences)
            if multi_pred != label:
                reward = -1
            else:
                reward = +1




        #################### PERFORMANCE DIFFERENCE AS REWARD ##########        
        """
        if action[1] == 1:
            self.make_query = True
            self.query()
            new_performance = self.get_performance(model)
            reward = new_performance - self.performance
            if new_performance != self.performance:
                #reward = 3.
                self.performance = new_performance
            # else:
                #reward = -1.
        else:
            reward = 0.
        """
        ################################################################


        # next frame
        next_frame = []
        is_terminal = False
        # the ending condition for one episode, return the accuracy of all the classifiers and the majority voting accuracy
        #if self.queried_times == self.budget or self.rounds == 5:
        if self.queried_times == self.budget or self.current_frame == len(self.train_x_all[0])-1:
            is_terminal = True
            self.terminal = True
            self.performance,_ = self.get_performance(model)
            self.reboot(model) 
            for i in range(len(model)):
                next_frame.append(self.train_x_all[i][self.order[self.current_frame]])
        # if after this round, and round < 5 the budget is not reached, start a new round

        #elif self.current_frame == len(self.train_x_all[0])-1 and self.queried_times < self.budget:
            # back to the start of the training data
        #    self.current_frame = 0
        #    for i in range(len(model)):
        #        next_frame.append(self.train_x_all[i][self.order[self.current_frame]])
        #    self.rounds += 1
        # not terminate, continuing this round
    
        else:
            self.terminal=False
            for i in range(len(model)):
                next_frame.append(self.train_x_all[i][self.order[self.current_frame+1]])
            self.current_frame += 1

        confidence_all = []
        predictions_all = []
        logits = []
        for i in range(len(model)):
            confidence_all.append(model[i].get_confidence(next_frame[i])[0])
            predictions = model[i].get_marginal(next_frame[i])
            logits.append(model[i].get_xlogits(next_frame[i], self.train_y_all[self.order[self.current_frame]]))
            for j in range(3): # 3 is hard coded as the number of classes
                predictions_all.append(predictions[0][j])
        predictions_all = np.array(predictions_all).reshape(1,3*len(model))
        confidence_all = np.array(confidence_all).reshape(len(model),)
        if len(self.feature) > 1:
            next_observation = [next_frame, confidence_all, predictions_all, np.squeeze(logits)]
        else:
            next_observation = [next_frame, confidence_all, predictions_all, [np.squeeze(logits)]]
        return reward, next_observation, is_terminal

    ### function to update the query set
    def query(self):
        if self.make_query == True:
            inputs = []
            for i in range(len(self.train_x_all)):
                inputs.append(self.train_x_all[i][self.order[self.current_frame]])

            labels = self.train_y_all[self.order[self.current_frame]]
            self.queried_times += 1
            self.queried_set_x.append(inputs)
            self.queried_set_y.append(labels)


    ### function to update the classifiers with the query set, return the accuracies of the classifiers
    def get_performance(self, tagger):            
        queried_set_xx = self.queried_set_x
        print(self.queried_times, ',' , self.current_frame, ',' , len(self.train_y_all))
        if self.queried_times != 0:
            if (self.queried_times % self.train_step == 0.0) or (self.current_frame == len(self.train_y_all)-1): #or self.rounds == 1:
                # model that update after each episode (ie after the budget is reached)
                print('update')
                indexs = [0]
                a = 0
                for i in range(len(tagger)):
                    a = a + tagger[i].n_input
                    indexs.append(a)

                if len(tagger) == 1:
                    tagger[i].train(np.squeeze(self.queried_set_x,axis=1), self.queried_set_y,self.feature[i])
                else:
                    if self.accum:
                        for i in range(self.budget*self.episode,self.budget*(self.episode+1)):
                            queried_set_xx[i] = np.concatenate((self.queried_set_x[i]),axis=1)
                    else:
                        for i in range(len(self.queried_set_x)):
                            queried_set_xx[i] = np.concatenate((self.queried_set_x[i]),axis=1)
                    queried_set_xx = np.array(queried_set_xx)
                    for i in range(len(tagger)):
                        tagger[i].train(queried_set_xx[:,:,indexs[i]:indexs[i+1]], self.queried_set_y,self.feature[i])
            
            performance = []
            predd_train = []
            y_majority_train = []
            for i in range(len(tagger)):
                performance.append(float("%.3f" % tagger[i].test(self.train_x_all[i], self.train_y_all)))
                predd_train.append(tagger[i].get_predictions(np.squeeze(self.train_x_all[i],axis=0)))

            predd_train = np.array(predd_train).T.tolist()

            for i in range(len(predd_train)):
                preds_train = Counter(predd_train[i])
                y_majority_train.append(preds_train.most_common(1)[0][0])


            accuracy_majority_train = sum(np.equal(self.train_y_all, y_majority_train))/len(self.train_y_all)
        else:
            performance = None
            accuracy_majority_train = None
        return performance, accuracy_majority_train
    
    ### move to util

    
    def reboot(self, model):
        # reboot everything to initial status
        # save in the csv file
        helper.write_csv_game(self,model)
        random.seed(self.cvit)
        random.shuffle(self.order)
        self.queried_times = 0
        self.terminal = False
        if not self.accum:
            self.queried_set_x = []
            self.queried_set_y = []
        self.current_frame = 0
        self.count_action0 = 0
        self.count_action1 = 0
        self.rAll = []
        self.episode += 1
        self.train_x_all =[]
        self.train_y_all = []
        self.test_x_all = []
        self.test_y_all = []
        self.rounds = 0
        for i in range(len(self.feature)):
            train_x, train_y = helper.load_traindata(self.feature[i],model[i],seed=self.episode)
            self.train_x_all.append(train_x)
            self.train_y_all = train_y
            test_x, test_y = helper.load_testdata(self.feature[i],model[i],seed=self.episode)
            self.test_x_all.append(test_x)
            self.test_y_all = test_y
        self.dev_x_all = self.test_x_all
        self.dev_y_all = self.test_y_all
        
        self.cross_counts_correct_train = np.zeros(shape=(len(self.feature)+1,len(self.feature)+2)).astype(int)
        self.cross_counts_incorrect_train = np.zeros(shape=(len(self.feature)+1,len(self.feature)+2)).astype(int)
        self.cross_counts_correct_test = np.zeros(shape=(len(self.feature)+1,len(self.feature)+2)).astype(int)
        self.cross_counts_incorrect_test = np.zeros(shape=(len(self.feature)+1,len(self.feature)+2)).astype(int)
