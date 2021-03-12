"""
Evolution of AutoBOT. Skrlj 2019
"""

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from scipy.sparse import hstack
import gzip
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import random_projection
try:
    import nltk
    nltk.data.path.append("nltk_data")
except Exception as es:
    import nltk
    print(es)
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import multiprocessing as mp
from nltk import pos_tag
import re
import string
from itertools import groupby
try:
    from nltk.tag import PerceptronTagger
except:
    def PerceptronTagger():
        return 0

#from sentence_embeddings import *
#from keyword_features import *
#from entity_detector import *

# from tax2vec import *

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn import pipeline
from sklearn.preprocessing import MaxAbsScaler,Normalizer
from sklearn import preprocessing
import numpy
numpy.random.seed()
from sklearn.feature_extraction.text import HashingVectorizer

def remove_punctuation(text):

    """
    This method removes punctuation
    """
    
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    return text

def remove_stopwords(text):

    """
    This method removes stopwords
    """
    
    stops = set(stopwords.words("english")).union(set(stopwords.words('spanish')))
    text = text.split()
    text = [x.lower() for x in text if x.lower() not in stops]
    return " ".join(text)

def remove_mentions(text, replace_token):

    """
    This method removes mentions (relevant for tweets)
    """
    
    return re.sub(r'(?:@[\w_]+)', replace_token, text)

def remove_URL(text):

    """
    This method removes mentions (relevant for tweets)
    """
    
    return re.sub(r'#URL#', '', text)

def remove_HASH(text):

    """
    This method removes mentions (relevant for tweets)
    """
    
    return re.sub(r'#HASHTAG#', '', text)


def remove_hashtags(text, replace_token):

    """
    This method removes hashtags
    """
    
    return re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", replace_token, text)

def remove_url(text, replace_token):

    """
    Removal of URLs
    """
    
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)

def get_affix(text):

    """
    This method gets the affix information
    """
    
    return " ".join([word[-4:] if len(word) >= 4 else word for word in text.split()])

def get_pos_tags(text):

    """
    This method yields pos tags
    """
    #tokenizer = TweetTokenizer()
    #tokens = tokenizer.tokenize(text)
    tokens = nltk.word_tokenize(text)
    tgx = " ".join([x[1] for x in pos_tag(tokens)])
    return tgx

def ttr(text):
    if len(text.split(" "))>1 and len(text.split()) > 0:
        return len(set(text.split()))/len(text.split())
    else:
        return 0

class text_col(BaseEstimator, TransformerMixin):

    """
    A helper processor class
    """
    
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]
    
#fit and transform numeric features, used in scikit Feature union
class digit_col(BaseEstimator, TransformerMixin):

    """
    Dealing with numeric features
    """
    
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['text', 'no_punctuation', 'no_stopwords', 'text_clean', 'affixes','pos_tag_seq']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        scaler = preprocessing.MinMaxScaler().fit(hd_searches)
        return scaler.transform(hd_searches)

def parallelize(data, method):
    
    """
    Helper method for parallelization
    """
    
    cores = mp.cpu_count()
    data_split = np.array_split(data, cores)
    pool = mp.Pool(cores)
    data = pd.concat(pool.map(method, data_split))
    pool.close()
    pool.join()
    return data
    
def build_dataframe(data_docs):

    """
    One of the core methods responsible for construction of a dataframe object.
    """
    df_data = pd.DataFrame({'text': data_docs})    
    df_data['no_punctuation'] = df_data['text'].map(lambda x: remove_punctuation(x))
    df_data['no_url'] = df_data['no_punctuation'].map(lambda x: remove_URL(x))
    df_data['no_hash'] = df_data['no_url'].map(lambda x: remove_HASH(x))
    df_data['no_stopwords'] = df_data['no_hash'].map(lambda x: remove_stopwords(x))
    df_data['text_clean'] = df_data['text']
    df_data['pos_tag_seq'] = df_data['text_clean'].map(lambda x: get_pos_tags(x))
    
    return df_data

class FeaturePrunner:

    """
    Core class describing sentence embedding methodology employed here.
    """
    
    def __init__(self, max_num_feat =  2048):

        self.max_num_feat = max_num_feat

    def fit(self,input_data, y = None):

        return self

    def transform(self, input_data):
        return input_data

    def get_feature_names(self):

        pass

def get_features(df_data, max_num_feat = 1000, labels = None):

    """
    Method that computes various TF-IDF-alike features.
    """

    tfidf_word_unigram = TfidfVectorizer(ngram_range=(1,2), max_features = max_num_feat)
    tfidf_char_unigram = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features = max_num_feat)
    hashing_vec = HashingVectorizer(n_features = max_num_feat)
    #  ('hash', pipeline.Pipeline([('s3', text_col(key='no_stopwords')), ('hash_tfidf', hashing_vec)]))
#    rex = KeywordFeatures(targets = labels)
    symbolic_features = [('word', pipeline.Pipeline([('s1', text_col(key='no_stopwords')), ('word_tfidf', tfidf_word_unigram)])),
                         ('char', pipeline.Pipeline([('s2', text_col(key='no_stopwords')), ('char_tfidf', tfidf_char_unigram)]))]
                
#    sentence_embedder_dm = documentEmbedder(max_features = 512, dm = 1)
    neural_features = []
    
#    neural_features = [('neural-doc2vec', pipeline.Pipeline([('s14', text_col(key='no_stopwords')), ('sentence_embedding_mean', sentence_embedder_dm)]))]

    features = symbolic_features + neural_features       
        
    feature_names = [x[0] for x in features]
    matrix = pipeline.Pipeline([
        ('union', FeatureUnion(
            transformer_list=features,n_jobs=-1)),
        ('normalize',Normalizer())])
    
    #print(matrix.fit_transform(df_data))
    try:
        data_matrix = matrix.fit_transform(df_data)
        tokenizer = matrix
        
    except Exception as es:        
        print(es, "Feature construction error.")
        tokenizer = None

    return tokenizer, feature_names, data_matrix
