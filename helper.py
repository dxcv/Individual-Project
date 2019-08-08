#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:13:56 2019

@author: orudovic
"""#give me a sec here

import numpy as np
import pickle as pkl
import os#
import pandas as pd
from collections import Counter
import csv
import random
from sklearn.metrics import f1_score

from keras.utils.np_utils import to_categorical




#os.environ['CUDA_VISIBLE_DEVICES']='0,1'  # Number of GPUs to run on
os.environ['CUDA_VISIBLE_DEVICES']='-1'

# depending on the naming of the data file, separate data file for different features
features = ['total','FACE', 'BODY', 'PHY', 'AUDIO']

def pred_maj(model, encoder, input_x):
    predictions = model.predict_classes(input_x)
    pred = np.argmax(to_categorical(predictions), axis = 1)
    pred = encoder.inverse_transform(pred)

    return pred





def load_traindata(feature,tagger,seed):
    dirName = "./data_all/training/train_{0}.pkl".format(features[feature])
    
    with open(dirName, 'rb') as f:
        dataset_x = pkl.load(f)
    with open("./data_all/training/train_label.pkl", 'rb') as f:
        dataset_y = np.array(pkl.load(f)).reshape(-1,)
    
    dataset_x = np.array(dataset_x).reshape(-1,tagger.n_steps, tagger.n_input)
    
    i_class0 = np.where(np.array(dataset_y)==0)[0]
    i_class1 = np.where(np.array(dataset_y)==1)[0]
    i_class2 = np.where(np.array(dataset_y)==2)[0]
    
    n_need = 500

    np.random.seed(3)
    add = np.random.randint(0,200)
    np.random.seed(seed+add)
    i_class0_upsampled = np.random.choice(i_class0, size=n_need, replace=True)
    i_class1_upsampled = np.random.choice(i_class1, size=n_need, replace=True)
    i_class2_downsampled = np.random.choice(i_class2, size=n_need, replace=True)
    
    total = np.concatenate((i_class0_upsampled, i_class1_upsampled, i_class2_downsampled))
    X_train_balanced = np.array(dataset_x)[total]
    y_train_balanced = np.array(dataset_y)[total]

    return list(X_train_balanced), list(y_train_balanced)

def load_testdata(feature,tagger,seed):
    dirName = "./data_all/testing/test_{0}.pkl".format(features[feature])
    
    with open(dirName, 'rb') as f:
        dataset_x = pkl.load(f)
    with open("./data_all/testing/test_label.pkl", 'rb') as f:
        dataset_y = np.array(pkl.load(f)).reshape(-1,)
    
    dataset_x = np.array(dataset_x).reshape(-1,tagger.n_steps, tagger.n_input)
    
    i_class0 = np.where(np.array(dataset_y)==0)[0]
    i_class1 = np.where(np.array(dataset_y)==1)[0]
    i_class2 = np.where(np.array(dataset_y)==2)[0]

    n_need = 500
    np.random.seed(3)
    add = np.random.randint(0,200)
    np.random.seed(0+add)
    i_class0_upsampled = np.random.choice(i_class0, size=n_need, replace=True)
    i_class1_upsampled = np.random.choice(i_class1, size=n_need, replace=True)
    i_class2_downsampled = np.random.choice(i_class2, size=n_need, replace=True)
    
    total = np.concatenate((i_class0_upsampled, i_class1_upsampled, i_class2_downsampled))
    X_train_balanced = np.array(dataset_x)[total]
    y_train_balanced = np.array(dataset_y)[total]

    return list(X_train_balanced), list(y_train_balanced)


def cross_count(env, preds_train_cross, preds_test_cross, y_multi_train, y_multi_test):
    for y_train, y_test, y_true in zip(y_multi_train, y_multi_test, env.train_y_all):
        if y_train == y_true:
            env.cross_counts_correct_train[-1][-2] += 1
        if y_test == y_true:
            env.cross_counts_correct_test[-1][-2] += 1

    for j_pred in range(env.cross_counts_correct_train.shape[0]-1):
        for y_multi, y_true, y_model in zip(y_multi_train, env.train_y_all, preds_train_cross[j_pred]):
            if y_multi == y_true and y_model == y_true:
                    env.cross_counts_correct_train[-1][j_pred] += 1
                    env.cross_counts_correct_train[j_pred][-2] += 1

            if y_multi == y_true and y_model != y_true:
                env.cross_counts_incorrect_train[-1][j_pred] += 1
            
            if y_model == y_true and y_multi != y_true:
                env.cross_counts_incorrect_train[j_pred][-2] += 1


    for i_pred in range(env.cross_counts_correct_train.shape[0]-1):
        for j_pred in range(env.cross_counts_correct_train.shape[0]-1):
            for y_model1, y_model2, y_true in zip(preds_train_cross[i_pred], preds_train_cross[j_pred], env.train_y_all):
                if y_model1 == y_true and y_model2 == y_true:
                    env.cross_counts_correct_train[i_pred][j_pred] += 1
                if y_model1 == y_true and y_model2 != y_true:
                    env.cross_counts_incorrect_train[i_pred][j_pred] += 1
        

    for j_pred in range(env.cross_counts_correct_test.shape[0]-1):
        for y_multi, y_true, y_model in zip(y_multi_test, env.test_y_all, preds_test_cross[j_pred]):
            if y_multi == y_true and y_model == y_true:
                    env.cross_counts_correct_test[-1][j_pred] += 1
                    env.cross_counts_correct_test[j_pred][-2] += 1
            if y_multi == y_true and y_model != y_true:
                env.cross_counts_incorrect_test[-1][j_pred] += 1
            
            if y_model == y_true and y_multi != y_true:
                env.cross_counts_incorrect_test[j_pred][-2] += 1


    for i_pred in range(env.cross_counts_correct_test.shape[0]-1):
        for j_pred in range(env.cross_counts_correct_test.shape[0]-1):
            for y_model1, y_model2, y_true in zip(preds_test_cross[i_pred], preds_test_cross[j_pred], env.test_y_all):
                if y_model1 == y_true and y_model2 == y_true:
                    env.cross_counts_correct_test[i_pred][j_pred] += 1
                if y_model1 == y_true and y_model2 != y_true:
                    env.cross_counts_incorrect_test[i_pred][j_pred] += 1



    env.cross_counts_correct_train[0][-1] = env.episode
    env.cross_counts_correct_test[0][-1] = env.episode
    env.cross_counts_incorrect_train[0][-1] = env.episode
    env.cross_counts_incorrect_test[0][-1] = env.episode

    f = open(env.csv_correct_train, "a")
    np.savetxt(f,  env.cross_counts_correct_train, delimiter=",")
    f.write('\n')
    f.close()
    f = open(env.csv_incorrect_train, "a")
    np.savetxt(f,  env.cross_counts_incorrect_train, delimiter=",")
    f.write('\n')
    f.close()
    f = open(env.csv_correct_test, "a")
    np.savetxt(f,  env.cross_counts_correct_test, delimiter=",")
    f.write('\n')
    f.close()
    f = open(env.csv_incorrect_test, "a")
    np.savetxt(f,  env.cross_counts_incorrect_test, delimiter=",")
    f.write('\n')
    f.close()

def voting_multi_instance(env, pred, confidences):
    if env.method == 'maj':
        preds = Counter(pred)
        pred_mode = preds.most_common(1)
        multi_pred = pred_mode[0][0]

    elif env.method == 'wei':
        pred_array = np.array(pred)
        weights = []
        for i in range(3):
            weights.append(sum(pred_array[np.where(pred_array==i)]))
        multi_pred = np.argmax(weights)

    elif env.method == 'conf':
        multi_pred = pred[np.argmax(np.array(confidences))]

    return multi_pred

def voting_multi_all(env, predd_train, predd_test, conf_train, conf_test):
    y_multi_train = []
    y_multi_test = []

    if env.method == 'maj':
        ### vote via majority
        for i in range(len(predd_train)):
            preds_train = Counter(predd_train[i])
            preds_test = Counter(predd_test[i])
            y_multi_train.append(preds_train.most_common(1)[0][0])
            y_multi_test.append(preds_test.most_common(1)[0][0])
        
    elif env.method == 'conf':
        ### vote via conf
        for i in range(len(predd_train)):
            y_multi_train.append(predd_train[i][np.argmax(np.array(conf_train[i]))])
        for i in range(len(predd_test)):
            y_multi_test.append(predd_test[i][np.argmax(np.array(conf_test[i]))])

    else:
        ### vote via weighted strategy            
        weights_train = np.zeros((len(predd_train),3))
        weights_test = np.zeros((len(predd_train),3))
        predd_train = np.array(predd_train)
        predd_test = np.array(predd_test)
        for i in range(len(predd_train)):
            for j in range(3):
                weights_train[i][j] = sum(predd_train[i][np.where(predd_train[i]==j)])

        for i in range(len(predd_test)):
            for j in range(3):
                weights_test[i][j] = sum(predd_test[i][np.where(predd_test[i]==j)])           

        y_multi_train = np.argmax(weights_train,axis=1)
        y_multi_test = np.argmax(weights_test,axis=1)

    return y_multi_train, y_multi_test

def write_csv_game(env, model):

    f = open(env.csvfile, "a")
    writer = csv.DictWriter(
        f, fieldnames=["episode_number", 
                       "accuracy_train",
                       "accuracy_test", 
                       "f1_train", 
                       "f1_test",
                       "accuracy_multi_train",
                       "f1_multi_train",
                       "accuracy_multi_test",
                       "f1_multi_test",
                       "count_0",
                       "count_1",
                       "reward_all",
                       "conf_train",
                       "conf_test"
                       ])
    if env.episode == 0:
        writer.writeheader()
    accuracy_test = []
    conf_train = []
    conf_all_train = []
    conf_all_test =[]
    conf_test = []
    f1_train = []
    f1_test = []
    predd_train = []
    predd_test = []


    for i in range(len(model)):

        accuracy_test.append(float("%.3f" % model[i].test(env.test_x_all[i], env.test_y_all)))
        conf_all_train.append(model[i].get_confidence(env.train_x_all[i]))
        conf_train.append(float("%.3f" % np.mean(model[i].get_confidence(env.train_x_all[i]))))
        conf_all_test.append(model[i].get_confidence(env.test_x_all[i]))
        conf_test.append(float("%.3f" % np.mean(model[i].get_confidence(env.test_x_all[i]))))
        f1_train.append(float("%.3f" % model[i].get_f1_score(env.train_x_all[i], env.train_y_all)))
        f1_test.append(float("%.3f" % model[i].get_f1_score(env.test_x_all[i], env.test_y_all)))
        predd_train.append(model[i].get_predictions(np.squeeze(env.train_x_all[i],axis=0)))
        predd_test.append(model[i].get_predictions(np.squeeze(env.test_x_all[i],axis=0)))

    predd_train = np.array(predd_train).T.tolist()
    predd_test = np.array(predd_test).T.tolist()
    conf_all_train = np.array(conf_all_train).T.tolist()
    conf_all_test = np.array(conf_all_test).T.tolist()

    y_multi_train, y_multi_test = voting_multi_all(env, predd_train, predd_test, conf_all_train, conf_all_test)
    #3y_multi_train = predd_train
    #y_multi_test = predd_test
    accuracy_multi_train = sum(np.equal(env.train_y_all, y_multi_train))/len(env.train_y_all)
    f1_multi_train = f1_score(env.train_y_all, y_multi_train, average='macro')
    accuracy_multi_test = sum(np.equal(env.test_y_all, y_multi_test))/len(env.test_y_all)
    f1_multi_test = f1_score(env.test_y_all, y_multi_test, average='macro')
    
    if len(model) > 1:
        cross_count(env, np.array(predd_train).T, np.array(predd_test).T, y_multi_train, y_multi_test)
    

    writer.writerow({"episode_number": env.episode, 
                     "accuracy_train": env.performance,  
                     "accuracy_test": accuracy_test, 
                     "f1_train": f1_train, 
                     "f1_test": f1_test,
                     "accuracy_multi_train": float("%.3f" % accuracy_multi_train),
                     "f1_multi_train": float("%.3f" % f1_multi_train),
                     "accuracy_multi_test": float("%.3f" % accuracy_multi_test),
                     "f1_multi_test": float("%.3f" % f1_multi_test),
                     "count_0": env.count_action0,
                     "count_1": env.count_action1,
                     "reward_all": np.sum(env.rAll),
                     "conf_train": conf_train,
                     "conf_test": conf_test})
    f.close()



def test_data(tagger, features_selected, student):


    field_inds = {'FACE': (0, 257), 
                'BODY': (257, 327),
                'PHY': (327, 354),
                'AUDIO': (354, 378),
                'CARS': (378, 393)} # we dont use this
    
    id_offset = 6
    fields = ('FACE', 'BODY', 'PHY', 'AUDIO', 'CARS')

    
    print(student)
    with open(student, 'rb') as f:
        raw = pkl.load(f)
        data_test = pd.DataFrame(raw)
        data_test.sort_values(5, ascending=True, inplace=True)
        data_test = np.array(data_test)

    raw_data_test = []
    label_test = []
    
    uniques = np.unique(data_test[:,3])
    df = pd.DataFrame(data_test)
    for unique in uniques:
        df_spec = df.loc[df[3] == unique]
        array_spec = np.array(df_spec)
        window_index = 0
        while window_index < len(array_spec)-30:
            sum_eng = 0
            if array_spec[window_index+30][5] - array_spec[window_index][5] == 30:
                for frame_raw in range(window_index, window_index+30, 3):
                    raw_data_test.append(array_spec[frame_raw])
                    sum_eng = sum_eng + array_spec[frame_raw][-1]
                #assign label based on 30 frames
                label_indicator = sum_eng/10.0
                if label_indicator < 0.5:
                    label_test.append(0)
                elif label_indicator >= 0.5 and label_indicator < 0.8:
                    label_test.append(1)
                else:
                    label_test.append(2)
                                        
                window_index = window_index + 5
            else:
                window_index += 1

    raw_data_test = np.array(raw_data_test).reshape((len(raw_data_test),402))

    #shape of 384
    test_feature_total = []
    test_feature_total.append(np.array(raw_data_test[:,6:378+6]).reshape(-1,10,378))
    for field in fields:
        start_col, end_col = field_inds[field]
        test_feature_total.append(np.array(raw_data_test[:, start_col + id_offset:end_col + id_offset]).reshape(-1,10,end_col-start_col))
    if len(features_selected) == 1:
        return [list(test_feature_total[features_selected[0]])], list(label_test)
    else:
        return list(test_feature_total[1:-1]), label_test

