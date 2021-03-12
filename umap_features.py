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
import umap
PICKLES_PATH = "umap_pickles"

def fit_space(texts, nfeat, dim, neigh, min_dist,  split = 'train'):
    if not split == 'train':
        return fit(texts)
    dataframe = build_dataframe(texts)
 
    tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = nfeat)
    reducer = umap.UMAP( n_neighbors=neigh, min_dist=min_dist, 
        n_components = min(dim, nfeat * len(feature_names)-1))#,        metric=metric )    
    data_matrix = reducer.fit_transform(data_matrix.toarray())
    if split == 'train':
        _export(tokenizer, reducer)
    return data_matrix

def _export(tokenizer, reducer):
    with open(os.path.join(PICKLES_PATH, "tokenizer_cv.pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join(PICKLES_PATH, "reducer_cv.pkl"),mode='wb') as f:
        pickle.dump(reducer,f)
        
def fit(X, model_path="."):
    df_final = build_dataframe(X)
    tokenizer,reducer = _import(lang="",path_in=model_path)
    matrix_form = tokenizer.transform(df_final)
    reduced_matrix_form = reducer.transform(matrix_form)
    return reduced_matrix_form 


def _import(lang='en',path_in="."):
    """Imports tokenizer,clf,reducer from param(path_in, default is ../models)"""
    tokenizer = pickle.load(open(os.path.join(PICKLES_PATH, "tokenizer_cv.pkl"),'rb'))
    reducer = pickle.load(open(os.path.join(PICKLES_PATH, "reducer_cv.pkl"),'rb'))
    return tokenizer,reducer
