# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 09:11:19 2021

@author: Bosec
"""


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

from sklearn.ensemble import RandomForestClassifier
import config
import logging
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import feature_construction
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import itertools

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
LM_types = ["lsa"]#,"stat"]#"distilbert-base-nli-mean-tokens","lsa","roberta-large-nli-stsb-mean-tokens","stat","xlm-r-large-en-ko-nli-ststb"]
from sklearn import preprocessing
def prepare(split, kg_types):
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
    return preprocessing.scale(dataz)

def prepare_dataset_snd(dataset, split, kg_types):    
    prefix = dataset+"/"+split
    train_matrix = prepare(prefix, kg_types)
    train_y = np.loadtxt(prefix+"/_ys.csv").astype(np.int64)
    train_y = train_y.tolist()
    return train_matrix, train_y
#outs = prepare("pan2020/train")
#print(outs.shape)
#train_nets("pan2020")           
    #Prepare the train data
def train(train_matrix, train_y, test_matrix, test_y):
    n_estimators = [10,20,50]# for x in np.linspace(start = 10, stop = 100, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [10,20,30]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    rf = RandomForestClassifier()
 
    gs1 = GridSearchCV(rf, random_grid, verbose = 1, n_jobs = 8,cv = 10, refit = True, scoring='accuracy')    
    gs1.fit(train_matrix, train_y)
    clf = gs1.best_estimator_    
    scores = cross_val_score(clf, train_matrix, train_y, n_jobs = 8, verbose = 1, cv=10, scoring='accuracy')
    acc_rf = scores.mean()
    logging.info("TRAIN RF 10fCV F1-score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    fitted = clf.fit(train_matrix, train_y)
    preds = fitted.predict(test_matrix)
    print("RF TEST: ",accuracy_score(test_y, preds), "F1_SCORE: ",f1_score(test_y, preds), "PRECISSION: ",precision_score(test_y, preds), "RECALL: ",recall_score(test_y, preds))
 
    
    
    
    
    parameters = {"loss":["hinge","log"],"penalty":["elasticnet"],"alpha":[0.01,0.001,0.0001,0.0005],"l1_ratio":[0.05,0.25,0.3,0.6,0.8,0.95],"power_t":[0.5,0.1,0.9]}
    svc = SGDClassifier()
    gs1 = GridSearchCV(svc, parameters, verbose = 1, n_jobs = 8,cv = 10, refit = True, scoring='accuracy')    
    gs1.fit(train_matrix, train_y)
    clf = gs1.best_estimator_    
    scores = cross_val_score(clf, train_matrix, train_y, n_jobs = 8, verbose = 1, cv=10, scoring='accuracy')
    acc_svm = scores.mean()
    logging.info("TRAIN SGD 10fCV F1-score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    fitted = clf.fit(train_matrix, train_y)
    preds = fitted.predict(test_matrix)
    print("SGD TEST: ",accuracy_score(test_y, preds), "F1_SCORE: ",f1_score(test_y, preds), "PRECISSION: ",precision_score(test_y, preds), "RECALL: ",recall_score(test_y, preds))
    
    params = {"C":[0.1,1,10,25,50,100,500,10000],"penalty":["l2"]}
    svc = LogisticRegression(max_iter = 100000,  solver="lbfgs")
    gs = GridSearchCV(svc, params, verbose = 1, n_jobs = 8,cv = 10, refit = True, scoring='accuracy')
    gs.fit(train_matrix, train_y)
    clf = gs.best_estimator_
    acc_lr = cross_val_score(clf, train_matrix, train_y, n_jobs = 8, verbose = 1, cv=10, scoring='accuracy')
    logging.info("TRAIN SGD 10fCV F1-score: %0.4f (+/- %0.4f)" % (acc_lr.mean(), acc_lr.std()))
        # Prepare output
    fitted = clf.fit(train_matrix, train_y)
    preds2 = fitted.predict(test_matrix)
    print("LR TEST: ",accuracy_score(test_y, preds2), "F1_SCORE: ",f1_score(test_y, preds2), "PRECISSION: ",precision_score(test_y, preds2), "RECALL: ",recall_score(test_y, preds2))
    if scores.mean()+scores.std() > acc_lr.mean()+acc_lr.std():
        return scores.mean()
    else:
        return acc_lr.mean()
org_kg_types = ["complex", "transe", "rotate", "quate", "distmult", "simple"]
kg_types_ = []
for i in range(len(org_kg_types)):
    for j in range(1,i+1):
        kg_types_.append(list(itertools.combinations(org_kg_types, j)))
#print(kg_types)
#org_kg_types = kg_types
dataz = {}
for kg in kg_types_:
    print(kg)
    for k in kg:
        print(k)
        train_matrix, train_y = prepare_dataset_snd("pan2020", "train", k)
        dev_matrix, dev_y = prepare_dataset_snd("pan2020", "dev",k)
        print(train_matrix.shape, dev_matrix.shape)
        train_matrix = np.vstack((np.asarray(train_matrix), np.asarray(dev_matrix)))
        print(type(train_y))
        train_y = train_y + dev_y
        test_matrix, test_y = prepare_dataset_snd("pan2020", "test",k)
        test_y = np.array(test_y)
        train_y = np.array(train_y)
        
        dataz[k] = train(train_matrix, np.array((train_y)), test_matrix, np.array((test_y)))

with open('dataz.pkl', 'wb') as f:
    pickle.dumps(dataz, f)