import numpy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
import time
import csv
from feature_construction import *
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle

try:
    import umap
except:
    pass
PICKLES_PATH = "lsa_pickles"
def learn_export(texts):
    nfeat = 5000
    dim = 512
    dataframe = build_dataframe(texts)
    tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = nfeat)
    reducer = TruncatedSVD(n_components = min(dim, nfeat * len(feature_names)-1))
    data_matrix = reducer.fit_transform(data_matrix)
    return data_matrix
    #_export(tokenizer, reducer)
    
def fit_space(texts, nfeat, dim, split = 'train', name ="asdas"):
    if not split == 'train':
        return fit(texts, name)
    dataframe = build_dataframe(texts)
    tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = nfeat)
    reducer = TruncatedSVD(n_components = min(dim, nfeat * len(feature_names)-1))
    data_matrix = reducer.fit_transform(data_matrix)
    if split == 'train':
        _export(tokenizer, reducer, name)
    return data_matrix

def _export(tokenizer, reducer, name = ""):
    with open(os.path.join(PICKLES_PATH, "tokenizer_"+name+".pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join(PICKLES_PATH, "reducer_"+name+".pkl"),mode='wb') as f:
        pickle.dump(reducer,f)
        
def fit(X,name):
    df_final = build_dataframe(X)
    tokenizer,reducer = _import(name)
    matrix_form = tokenizer.transform(df_final)
    reduced_matrix_form = reducer.transform(matrix_form)
    return reduced_matrix_form 


def _import(name):
    """Imports tokenizer,clf,reducer from param(path_in, default is ../models)"""
    tokenizer = pickle.load(open(os.path.join(PICKLES_PATH, "tokenizer_"+name+".pkl"),'rb'))
    reducer = pickle.load(open(os.path.join(PICKLES_PATH, "reducer_"+name+".pkl"),'rb'))
    return tokenizer,reducer
