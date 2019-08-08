import numpy as np
import pickle as pkl
import pandas as pd

def load_train_test_data(type_data, feature):
    
    field_inds = {'FACE': (0, 257), 
    			'BODY': (257, 327),
    			'PHY': (327, 354),
    			'AUDIO': (354, 378),
    			'CARS': (378, 393)} # we dont use this
    
    id_offset = 6
    y_offset = 3
    fields = ('FACE', 'BODY', 'PHY', 'AUDIO', 'CARS')
    
    
    
    
    test_child = []
    train_child = []
    
    with open('/data/orudovic/v5/test_child.txt') as f:
    	for line in f.readlines():
    		l = line.split()[0]
    		test_child.append(l)
    
    with open('/data/orudovic/v5/train_child.txt') as f:
    	for line in f.readlines():
    		l = line.split()[0]
    		train_child.append(l)
    
    
    # load train data
    data_train = list()
    added_train = set()  # set of all filenames that have already been added to data
    data_test = list()
    added_test = set()
    
    for file in train_child:
        with open(file, 'rb') as f:
            raw = pkl.load(f)
            raw = pd.DataFrame(raw)
            raw.sort_values(5, ascending=True, inplace=True)
            data_train.append(np.array(raw))
            added_train.add(file)
    
    for file in test_child:
        with open(file, 'rb') as f:
            raw = pkl.load(f)
            raw = pd.DataFrame(raw)
            raw.sort_values(5, ascending=True, inplace=True)
            data_test.append(np.array(raw))
            added_test.add(file)
    
    # should not concatenate
    '''
    # merge imported arraies into large array
    raw_data_train = np.concatenate(data_train, axis=0)
    raw_data_test = np.concatenate(data_test, axis=0)
    '''
    
    #########################################################
    # TODO: use 30 frame to pick
    raw_data_train = []
    label_train = []
    raw_data_test = []
    label_test = []
    
    for child in range(len(data_train)):
        uniques = np.unique(data_train[child][:,3])
        df = pd.DataFrame(data_train[child])
        for unique in uniques:
            df_spec = df.loc[df[3] == unique]
            array_spec = np.array(df_spec)
            window_index = 0
            while window_index < len(array_spec)-30:
                sum_eng = 0
                if array_spec[window_index+30][5] - array_spec[window_index][5] == 30:
                    for frame_raw in range(window_index, window_index+30, 3):
                        raw_data_train.append(array_spec[frame_raw])
                        sum_eng = sum_eng + array_spec[frame_raw][-1]
                    #assign label based on 30 frame
                    label_indicator = sum_eng/10.0  
                    if label_indicator < 0.5:
                        label_train.append(0)
                    elif label_indicator >= 0.5 and label_indicator < 0.8:
                        label_train.append(1)
                    else:
                        label_train.append(2)
                        
                    window_index = window_index + 30
                else:
                    window_index += 1
    raw_data_train = np.array(raw_data_train).reshape((len(raw_data_train),402))
    
    
    
    for child in range(len(data_test)):
        uniques = np.unique(data_test[child][:,3])
        df = pd.DataFrame(data_test[child])
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
                                            
                    window_index = window_index + 30
                else:
                    window_index += 1
    raw_data_test = np.array(raw_data_test).reshape((len(raw_data_test),402))
    
    ##########################################################        
    
    
    train_feature_total = pd.DataFrame(raw_data_train[:,6:378+6])
    test_feature_total = pd.DataFrame(raw_data_test[:,6:378+6])
    train_feature_total.to_pickle('train_total.pkl')
    test_feature_total.to_pickle('test_total.pkl')
    pd.DataFrame(label_train).to_pickle('train_label.pkl')
    pd.DataFrame(label_test).to_pickle('test_label.pkl')
    

    train_feature_balanced, train_label_balanced = balance_data(train_feature_total, label_train)
    test_feature_balanced, test_label_balanced = balance_data(test_feature_total, label_test)

        
    if type_data == 'total':    
        return train_feature_balanced, train_label_balanced, test_feature_balanced, test_label_balanced
    
    if feature != None:
        feature_data_train = list()
        feature_data_train.append(raw_data_train[:, 0:id_offset])  # append id columns
        for field in fields:
        	start_col, end_col = field_inds[field]
        	feature_data_train.append(raw_data_train[:, start_col + id_offset:end_col + id_offset])
        feature_data_train.append(raw_data_train[:, -y_offset:])  # append labels
        
        feature_data_test = list()
        feature_data_test.append(raw_data_test[:, 0:id_offset])  # append id columns
        for field in fields:
        	start_col, end_col = field_inds[field]
        	feature_data_test.append(raw_data_test[:, start_col + id_offset:end_col + id_offset])
        feature_data_test.append(raw_data_test[:, -y_offset:])  # append labels
        
        return_feature_train = []
        return_feature_test = []
        for i in feature:
            return_feature_train.append(feature_data_train[i+1])
            return_feature_test.append(feature_data_test[i+1])



def balance_data(dataset_x, dataset_y, seed=0):
    i_class0 = np.where(np.array(dataset_y)==0)[0]
    i_class1 = np.where(np.array(dataset_y)==1)[0]
    i_class2 = np.where(np.array(dataset_y)==2)[0]
    
    n_need = 500
    np.random.seed(seed)
    i_class0_upsampled = np.random.choice(i_class0, size=n_need, replace=True)
    i_class1_upsampled = np.random.choice(i_class1, size=n_need, replace=True)
    i_class2_downsampled = np.random.choice(i_class2, size=n_need, replace=True)
    
    total = np.concatenate((i_class0_upsampled, i_class1_upsampled, i_class2_downsampled))
    X_train_balanced = np.array(dataset_x)[total]
    y_train_balanced = np.array(dataset_y)[total]
    return list(X_train_balanced), list(y_train_balanced)
'''


for i in range(4):
    save_data = pd.DataFrame(feature_data_train[i+1])
    save_data.to_pickle('train_'+fields[i]+'.pkl')

for i in range(4):
    save_data = pd.DataFrame(feature_data_test[i+1])
    save_data.to_pickle('test_'+fields[i]+'.pkl')

'''
