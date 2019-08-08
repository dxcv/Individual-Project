#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:45:05 2019
UPDATED: Sat May 25
ADDED FUNCTIONS "conservative sampling", "Jaccord similarity"
@author: Beryl
"""
import sys
import os
#os.environ['CUDA_VISIBLE_DEVICES']='0,1'  # Number of GPUs to run on
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import csv
import numpy as np
import helper
import tensorflow as tf
import random
from tagger import Tagger
from sklearn.metrics import f1_score
from diversity import diversitySampling
from collections import Counter



AGENT = "LSTMQ"
MAX_EPISODE = 100

##################################################################
# changable variables
BUDGETS = [5, 10, 20, 50, 100]
#BUDGETS = [100, 300]
NITERS = [10] # number of epochs 
POLYS = [False]
CONTENTS = [False]
CUMS = [False] #cumulative
FEATURES = [[0],[1],[2],[3],[4],[1,2,3,4]] # WHAT ABOUT THE OTHER FEATURE COMBINATIONS?
SAMPLE_METHOD = ['rs', 'us', 'ds', 'cs', 'lc'] #random, uncertainty, diversity, conservative, least confidence
VOTE_METHOD = ['maj', 'wei', 'conf']
#FEATURES = [[0], [1,2,3,4]]
#################################################################
#feature shape
FEATURE_SHAPE = [[378,10],[257,10],[70,10],[27,10],[24,10]]
#FEATURE = 'ALL','FACE', 'BODY', 'PHY', 'AUDIO']

def data_generation(model, feature_number):

    train_x_all = []
    test_x_all = []
    
    for i in range(len(model)):
        print("Loading data for feature {0}..".format(feature_number[i]))
        train_x, train_y = helper.load_traindata(feature_number[i],model[i],seed=0)
        train_x_all.append(train_x)
        train_y_all = train_y
        test_x, test_y = helper.load_testdata(feature_number[i],model[i],seed=0)
        test_x_all.append(test_x)
        test_y_all = test_y

    
    train = [train_x_all, train_y_all]
    test = [test_x_all, test_y_all]

    return train, test
###############################################################################

########################### RANDOM SAMPLING ###################################
def random_sampling(feature_now, model_ver, budget):
    csvfile = 'records_rs/'+model_ver+'.csv'
    model_selected = []

    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                                n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],feature_number=i)
        model.train([],[],feature_number=i)
        model_selected.append(model)
        
    train_data, test_data = data_generation(model_selected, feature_now)
    train_x_all = train_data[0]
    test_x_all = test_data[0]
    train_y_all = train_data[1]
    test_y_all = test_data[1]
    
    order = np.linspace(0, len(train_x_all[0])-1, len(train_x_all[0]))
    order = order.astype(int)

    episode = 1
    print(">>>>>> Playing game ..")
    while episode <= MAX_EPISODE:
        
        sample_N = min(budget*4,len(train_y_all))
        order = np.linspace(0, sample_N-1, sample_N)
        order = order.astype(int)
         
        budget = min(budget,len(order))
        random.seed(episode)
        random.shuffle(order)
        queried_indexs = random.sample(list(order), budget)
        for i in range(len(model_selected)):
            model_selected[i].train(np.array(train_x_all[i])[queried_indexs], np.array(train_y_all)[queried_indexs], feature_now[i])
        print(">>>>>> Terminate ...")
        write_csv(episode, csvfile, model_selected, train_x_all, test_x_all, train_y_all, test_y_all)
        episode = episode+1
###############################################################################



######################### UNCERTAINTY SAMPLING ################################
def uncertainty_sampling(feature_now, model_ver, budget):
    csvfile = 'records_us/'+model_ver+'.csv'
    model_selected = []

    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                                n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],feature_number=i)
        model.train([],[],feature_number=i)
        model_selected.append(model)
        
    train_data, test_data = data_generation(model_selected, feature_now)
    train_x_all = train_data[0]
    test_x_all = test_data[0]
    train_y_all = train_data[1]
    test_y_all = test_data[1]

    episode = 1
    print(">>>>>> Playing game ..")
    while episode <= MAX_EPISODE:
        # compute uncertaity, which is 1-confidence:
        sample_N = min(budget*4,len(train_y_all))
        
        N = len(train_y_all)
        budget = min(budget,N)
        uncertainty = np.zeros((sample_N,))
        ones = np.ones((sample_N,))
        
        for i in range(len(model_selected)):
            uncertainty = uncertainty + (ones - model_selected[i].get_confidence(list(train_x_all[i][:sample_N])))
            print(uncertainty)
        queried_indexs = sorted(range(len(uncertainty)), key=lambda i: uncertainty[i])[-budget:]
        print('top uncertainties found')
        for i in range(len(model_selected)):
            model_selected[i].train(np.array(train_x_all[i])[queried_indexs], np.array(train_y_all)[queried_indexs], feature_now[i])
        print(">>>>>> Terminate ...")
        write_csv(episode, csvfile, model_selected, train_x_all, test_x_all, train_y_all, test_y_all)
        episode = episode+1
###############################################################################



######################### COSERVATIVE SAMPLING ################################       
def conservative_sampling(feature_now, model_ver, budget):
    csvfile = 'records_cs/'+model_ver+'.csv'
    model_selected = []

    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                                n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],feature_number=i)
        model.train([],[],feature_number=i)
        model_selected.append(model)
        
    train_data, test_data = data_generation(model_selected, feature_now)
    train_x_all = train_data[0]
    test_x_all = test_data[0]
    train_y_all = train_data[1]
    test_y_all = test_data[1]

    episode = 1
    print(">>>>>> Playing game ..")
    while episode <= MAX_EPISODE:
        # compute uncertaity, which is 1-confidence:

        sample_N = min(budget*4,len(train_y_all))
        
        N = len(train_y_all)
        budget = min(budget,N)
        confidence = []
        conf_diff = np.zeros((sample_N,))
        
        for i in range(len(model_selected)):
            confidence.append(model_selected[i].get_confidence(list(train_x_all[i][:sample_N])))
        # the max indecies
        ind_max = np.argmax(confidence, axis=0)
        # the min indecies
        ind_min = np.argmin(confidence, axis=0)
        for i in range(sample_N):
            conf_diff[i] = confidence[ind_max[i]][i]-confidence[ind_min[i]][i]
        queried_indexs = sorted(range(len(conf_diff)), key=lambda i: conf_diff[i])[:budget]
        
        print('top uncertainties found')
        for i in range(len(model_selected)):
            model_selected[i].train(np.array(train_x_all[i])[queried_indexs], np.array(train_y_all)[queried_indexs], feature_now[i])
        print(">>>>>> Terminate ...")
        write_csv(episode, csvfile, model_selected, train_x_all, test_x_all, train_y_all, test_y_all)
        episode = episode+1
###############################################################################
        
       
        
############################## DIVERSITY SAMPLING #############################
def diversity_sampling(feature_now, model_ver, budget):
    csvfile = 'records_us/'+model_ver+'.csv'
    model_selected = []

    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                                n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],feature_number=i)
        model.train([],[],feature_number=i)
        model_selected.append(model)
        
    train_data, test_data = data_generation(model_selected, feature_now)
    train_x_all = train_data[0]
    train_y_all = train_data[1]
    test_x_all = test_data[0]
    test_y_all = test_data[1]
    episode = 1
    print(">>>>>> Playing game ..")
    while episode <= MAX_EPISODE:
        sample_N = min(budget*4,len(train_y_all))
        
        N = len(train_y_all)
        budget = min(budget,N)
        
        s = diversitySampling(train_x_all[:,:sample_N], pool = [], budget = budget)
        s.updateCplus()
        queried_indexs = s.newind
        for i in range(len(model_selected)):
            model_selected[i].train(np.array(train_x_all[i])[queried_indexs], np.array(train_y_all)[queried_indexs], feature_now[i])
        print(">>>>>> Terminate ...")
        write_csv(episode, csvfile, model_selected, train_x_all, test_x_all, train_y_all, test_y_all)
        episode = episode+1
###############################################################################


        
############################# LEAST CONFIDENT #################################
def least_confident(feature_now, model_ver, budget):
    
        
        
        
        
###############################################################################



###############################################################################
      
def write_csv(episode, csvfile, model, train_x_all, test_x_all, train_y_all, test_y_all):

        f = open(csvfile, "a")
        writer = csv.DictWriter(
            f, fieldnames=["episode_number", 
                           "accuracy_train",
                           "accuracy_test", 
                           "f1_train", 
                           "f1_test",
                           "accuracy_majority_train",
                           "f1_majority_train",
                           "accuracy_majority_test",
                           "f1_majority_test",
                           "conf_train",
                           "conf_test"])
        if episode == 1:
            writer.writeheader()
        accuracy_train = []
        accuracy_test = []
        conf_train = []
        conf_test = []
        f1_train = []
        f1_test = []
        predd_train = []
        predd_test = []
        y_majority_train = []
        y_majority_test = []

        for i in range(len(model)):

            accuracy_train.append(float("%.3f" % model[i].test(train_x_all[i], train_y_all)))
            accuracy_test.append(float("%.3f" % model[i].test(test_x_all[i], test_y_all)))
            conf_train.append(float("%.3f" % np.mean(model[i].get_confidence(train_x_all[i]))))
            conf_test.append(float("%.3f" % np.mean(model[i].get_confidence(test_x_all[i]))))
            f1_train.append(float("%.3f" % model[i].get_f1_score(train_x_all[i], train_y_all)))
            f1_test.append(float("%.3f" % model[i].get_f1_score(test_x_all[i], test_y_all)))
            predd_train.append(model[i].get_predictions(np.squeeze(train_x_all[i],axis=0)))
            predd_test.append(model[i].get_predictions(np.squeeze(test_x_all[i],axis=0)))

        predd_train = np.array(predd_train).T.tolist()
        predd_test = np.array(predd_test).T.tolist()

        for i in range(len(predd_train)):
            preds_train = Counter(predd_train[i])
            preds_test = Counter(predd_test[i])
            y_majority_train.append(preds_train.most_common(1)[0][0])
            y_majority_test.append(preds_test.most_common(1)[0][0])
        accuracy_majority_train = sum(np.equal(train_y_all, y_majority_train))/len(train_y_all)

        f1_majority_train = f1_score(train_y_all, y_majority_train, average='macro')
        accuracy_majority_test = sum(np.equal(test_y_all, y_majority_test))/len(test_y_all)
        f1_majority_test = f1_score(test_y_all, y_majority_test, average='macro')
        
            
        writer.writerow({"episode_number": episode, 
                         "accuracy_train": accuracy_train,  
                         "accuracy_test": accuracy_test, 
                         "f1_train": f1_train, 
                         "f1_test": f1_test,
                         "accuracy_majority_train": float("%.3f" % accuracy_majority_train),
                         "f1_majority_train": float("%.3f" % f1_majority_train),
                         "accuracy_majority_test": float("%.3f" % accuracy_majority_test),
                         "f1_majority_test": float("%.3f" % f1_majority_test),
                         "conf_train": conf_train,
                         "conf_test": conf_test})

        print('csv saved')
        f.close()

def main():
    global AGENT, MAX_EPISODE, BUDGET, MODEL_VER, FEATURE, FEATURE_SHAPE, CONTENT, CUM, NITER, POLYS, SAMPLE_METHOD, VOTE_METHOD
    
    for budget in BUDGETS:
        BUDGET=budget
        for niter in NITERS:
            NITER=niter
            for content in CONTENTS:
                CONTENT = content
                for cum in CUMS:
                    CUM = cum
                    for feature in FEATURES:
                        FEATURE = feature
                        for method in SAMPLE_METHOD:
                            for vote in VOTE_METHOD:
                                MODEL_VER_0 = '{0}_{1}/model_it_{2}_budget_{3}_content_{4}_cum_{5}'.format(method, vote, NITER, BUDGET, int(CONTENT), int(CUM))
                                
                                s=[0,0,0,0]
                                fvar = '_feature'
                                for i in range(np.shape(FEATURE)[0]):
                                    if FEATURE[i]:
                                        s[FEATURE[i]-1]=1
                                for i in range(np.shape(s)[0]):
                                    fvar = fvar+'_{0}'.format(s[i])
        
                                MODEL_VER_0 = MODEL_VER_0 +str(fvar)
        
                                if CONTENT: 
                                    POLY=False
                                    MODEL_VER = MODEL_VER_0 + '_poly_{0}'.format(int(POLY))
                                    print('this iter', MODEL_VER)
                                    if method == 'rs':
                                        random_sampling(feature, MODEL_VER, BUDGET)
                                    elif method == 'us':
                                        uncertainty_sampling(feature, MODEL_VER, BUDGET)
                                    elif method == 'cs':
                                        conservative_sampling(feature, MODEL_VER, BUDGET)
                                    elif method == 'ds':
                                        diversity_sampling(feature, MODEL_VER, BUDGET)
                                    else:
                                        least_confident(feature, MODEL_VER, BUDGET)
                                    tf.reset_default_graph()
                                else:
                                    for poly in POLYS:
                                        POLY = poly
                                        MODEL_VER = MODEL_VER_0 + '_poly_{0}'.format(int(POLY))
                                        print('this iter', MODEL_VER)
                                        if method == 'rs':
                                            random_sampling(feature, MODEL_VER, BUDGET)
                                        elif method == 'us':
                                            uncertainty_sampling(feature, MODEL_VER, BUDGET)
                                        elif method == 'cs':
                                            conservative_sampling(feature, MODEL_VER, BUDGET)
                                        elif method == 'ds':
                                            diversity_sampling(feature, MODEL_VER, BUDGET)
                                        else:
                                            least_confident(feature, MODEL_VER, BUDGET)
                                        tf.reset_default_graph()

                        
                


if __name__ == '__main__':
    main()
