# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:11:14 2020

@author: Bosec
"""
from torch.utils.data import DataLoader

import os
import numpy as np
import pandas as pd
from model import train_NN, predict, fit
from model_helper import ColumnarDataset
from sklearn import preprocessing



"""
def prepare_dataset(data, prep = prepare_texts_5net):
    train_texts = data["text_a"].to_list()
    train_y = data['label'].to_list()
    train_matrix = prep(train_texts) 
    
    #del train_texts
    #train_matrix = pd.DataFrame(train_matrix)
    print(train_matrix)
    train_dataset = ColumnarDataset(train_matrix, train_y)
    return train_dataset, train_matrix.shape[1]      
def fit_space(data, prep = prepare_texts_berts):
    train_texts = data
    train_y = [0] * len(data)
    train_matrix = prep(train_texts) 
    
    #del train_texts
    #train_matrix = pd.DataFrame(train_matrix)
    print(train_matrix)
    train_dataset = ColumnarDataset(train_matrix, train_y)
    test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return test_loader

import torch 
def train_nets(train_data, dev_data , test_data):
    train_dataset, dims = prepare_dataset(train_data, prep = prepare_texts_berts)
    valid_dataset, dims = prepare_dataset(dev_data, prep = prepare_texts_berts)   
    test_dataset, dims = prepare_dataset(test_data, prep = prepare_texts_berts)
    for lr in [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        for p in [0.1, 0.3, 0.5, 0.7]:
            print(lr, p)
            net = train_NN(train_dataset, valid_dataset, dims ,max_epochs=100, batch_size = 32, lr = lr, dropout = p)
            predict(test_dataset, net)
            torch.save(net.state_dict(), os.path.join("pickles","t2v_bert_lsa_"+str(lr)+"_"+str(p)+".pymodel"))

"""
import torch 

def prepare_dataset(dataset, split):    
    prefix = dataset+"/"+split
    train_matrix = prepare(prefix)
    train_y = np.loadtxt(prefix+"/_ys.csv").astype(np.int64)
    train_y = train_y.tolist()
    train_dataset = ColumnarDataset(train_matrix, train_y)
    return train_dataset, train_matrix.shape[1]     

def train_nets(dataset):
    train_dataset, dims = prepare_dataset(dataset, split = 'train')
    valid_dataset, dims = prepare_dataset(dataset, split = 'dev') 
    test_dataset, dims = prepare_dataset(dataset, split = 'test')
    for lr in [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
            for e_max in [12,13,14]:
                net = train_NN(train_dataset, valid_dataset, dims, e_max = e_max, max_epochs=1000, batch_size = 32, lr = lr, dropout = p)
                predict(test_dataset, net)
            #torch.save(net.state_dict(), os.path.join("pickles","t2v_bert_lsa_"+str(lr)+"_"+str(p)+".pymodel"))



kg_types = ["complex", "transe", "rotate", "quate", "distmult", "simple"]
LM_types = ["distilbert-base-nli-mean-tokens","lsa","roberta-large-nli-stsb-mean-tokens","stat","xlm-r-large-en-ko-nli-ststb"]
def prepare(split):
    dataz = None
    first = True
    for lm in LM_types:
        path = split+"/"+lm+".csv"
        df = pd.read_csv(path,header=None).to_numpy()
        if first:
            dataz = df
            first = False
        else:
            dataz = np.hstack((dataz, df))
    for kg in kg_types:
        path = split+"/"+kg+".csv"
        df = pd.read_csv(path,header=None).to_numpy()
        dataz = np.hstack((dataz, df))
    return dataz

def prepare_dataset_snd(dataset, split):    
    prefix = dataset+"/"+split
    train_matrix = prepare(prefix)
    train_y = np.loadtxt(prefix+"/_ys.csv").astype(np.int64)
    train_y = train_y.tolist()
    return train_matrix, train_y
#outs = prepare("pan2020/train")
#print(outs.shape)
#train_nets("pan2020")           
    #Prepare the train data
    train_matrix = prepare_text(X, model) 
       
    parameters = {"loss":["hinge","log"],"penalty":["elasticnet"],"alpha":[0.01,0.001,0.0001,0.0005],"l1_ratio":[0.05,0.25,0.3,0.6,0.8,0.95],"power_t":[0.5,0.1,0.9]}
    svc = SGDClassifier()
    gs1 = GridSearchCV(svc, parameters, verbose = 1, n_jobs = 8,cv = 10, refit = True)     
    gs1.fit(train_matrix, final_y)
    clf = gs1.best_estimator_    
    scores = cross_val_score(clf, train_matrix, final_y, n_jobs = 8, verbose = 1, cv=10, scoring='f1_macro')
    acc_svm = scores.mean()
    logging.info("TRAIN SGD 10fCV F1-score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                 
    parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l1","l2"]}
    """
    lr_learner = LogisticRegression(max_iter = 100000,  solver="saga")
    gs = GridSearchCV(lr_learner, parameters, verbose = 1, n_jobs = 8,cv = 10, refit = True)
    gs.fit(train_matrix, final_y)
    clf = gs.best_estimator_
    acc_lr = cross_val_score(clf, train_matrix, y, n_jobs = 8, verbose = 1, cv=10, scoring='f1_macro')
    logging.info("TRAIN SGD 10fCV F1-score: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    
    if acc_svm > acc_lr:
        clf = clf1
    """
    
    # Prepare output
    #fitted = clf.fit(train_matrix, train_y)
    with open(os.path.join(config.PICKLES_PATH, model + "_cv_clf.pkl"), "wb") as f:
        pickle.dump(clf, f)
    return
    return clf
train_matrix, train_y = prepare_dataset_snd("pan2020", "train")
dev_matrix, dev_y = prepare_dataset_snd("pan2020", "dev")
test_matrix, test_y = prepare_dataset_snd("pan2020", "test")

