# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 07:49:33 2020

@author: Bosec
"""

from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from nltk import word_tokenize
from string import punctuation
def get_features():
    f = ['min_len','max_len', 'upper', 'lower', 'mean', 'digits', 'letters', 'spaces' , 'punct',  'hash']
def count_vowels(text):
    vowels = "aeiou"
    v_dict = {}
    for v in vowels:
        v_dict[v] = text.count(v)
    return v_dict

def count_hashes(text):
    return text.count('#')

def count_links(text):
    pass

def count_word_based(text):
    word_stats = {}
    stat = [] 
    upper = 0
    lower = 0
    min_len = 1000
    max_len = -min_len
    for word in word_tokenize(text):
        lower = lower + 1
        if word[0].isupper():
            upper = upper + 1
            lower = lower - 1
        stat.append(len(word))
        max_len = max(max_len,len(word))
        min_len = min(min_len,len(word))
    stat = np.array(stat)    
    word_stats['min_len'] = min_len
    word_stats['max_len'] = max_len
    word_stats['upper'] = upper
    word_stats['lower'] = lower
    word_stats['mean'] = np.mean(stat)
    word_stats['std'] = np.std(stat)
    return np.array(list(word_stats.values()))

def count_char_based(text):
    char_stats = {'digits':0, 'letters':0, 'spaces':0 , 'punct': 0}    
    for c in text:
        if c.isdigit():
            char_stats['digits'] = char_stats['digits'] + 1
        elif c.isalpha():
            char_stats['letters'] = char_stats['letters'] + 1
        elif c.isspace():
            char_stats['spaces'] = char_stats['spaces'] + 1
        elif c in punctuation:
            char_stats['punct'] = char_stats['punct'] + 1
    #char_stats['other'] = len(text) - char_stats['digits'] - char_stats['letters'] - char_stats['spaces'] - char_stats['punct']
    char_stats.update(count_vowels(text))
    char_stats['hash'] = count_hashes(text)
    return np.array(list(char_stats.values()))


def build_features(texts):
    df_data = {} 
    df_data['w_based'] = np.array(list(map(count_word_based, texts)))
    df_data['c_based'] = np.array(list(map(count_char_based, texts)))
    feature_mat = np.concatenate((df_data['w_based'], df_data['c_based']), axis=1)
    centered =  preprocessing.scale(feature_mat)
    return centered


def fit_space(texts):
    df_data = {} 
    df_data['w_based'] = np.array(list(map(count_word_based, texts)))
    df_data['c_based'] = np.array(list(map(count_char_based, texts)))
    feature_mat = np.concatenate((df_data['w_based'], df_data['c_based']), axis=1)
    centered =  preprocessing.scale(feature_mat)
    return centered