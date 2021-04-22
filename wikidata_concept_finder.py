#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Oct 18 07:06:38 2020

@author: daskalot

"""

#from doc_embedders import TfIdfTransformer as tfidf
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn import preprocessing
import graphvite as gv
import numpy as np
from tqdm import tqdm
import pickle
import os
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer() 
lemma = lemmatizer.lemmatize
my_stop_words = [lemma(t) for t in stopwords.words('english')]

def remove_punctuation(text):  
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    return text

def tokenize(text):
    no_punct = remove_punctuation(text)
    stems = [lemma(t) for t in word_tokenize(no_punct) if not t in my_stop_words ]
    return stems

def naive_terms(texts, n=2):
    print("Term finding started.")    
    vectorizer = CountVectorizer(tokenizer = tokenize, strip_accents = 'ascii', ngram_range = (1,n))#, stop_words=my_stop_words)
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.inverse_transform(X)
    print("Term finding finished.")
    return terms

def wikify(unique_terms, model_name="transe"):
    present_terms = {}
    print("Wikifying started.")
    with open("kg/"+model_name+".pkl", "rb") as fin:
        model = pickle.load(fin)
        
    alias2entity = gv.dataset.wikidata5m.alias2entity
    entity2id = model.graph.entity2id
    entity_embeddings = model.solver.entity_embeddings
    
    relation2id = model.graph.relation2id
    relation_embeddings = model.solver.relation_embeddings
    alias2relation = gv.dataset.wikidata5m.alias2relation
    
    for term in unique_terms:
            try:
                if term in alias2entity:
                    present_terms[term] = entity_embeddings[entity2id[alias2entity[term]]]
            except: 
                    pass
            try:
                if term in alias2relation:
                    present_terms[term] = relation_embeddings[relation2id[alias2relation[term]]]
            except:
                pass
    print("Wikifying finished.")
    return present_terms

def naive_embedd(docs, ngrams = 2, model_name = "transe"):
    term_candidates = naive_terms(docs)
    with open("kg/wikidata_aliases.pkl", "rb") as fin:
        entities = pickle.load(fin)
    with open("kg/wikidata_relations.pkl", "rb") as fin:
        relations = pickle.load(fin)
    unique_terms = set(x for c in term_candidates for x in c)
    unique_present = set(unique_terms).intersection(entities)
    doc2terms = []

    for doc in term_candidates:
        curr_terms = []
        for term in doc:
            if term in unique_present:
                curr_terms.append(term)
        doc2terms.append(curr_terms)

    present_terms = wikify(unique_present, model_name)
    print("Embedding started.")
    doc_kg_embs = []
    final_terms = []
    for doc in doc2terms:
        term2vec = {}
        for term in doc:
            if term in present_terms:
                term2vec[term] = present_terms[term]
        doc_kg_emb = make_embedding(term2vec)
        doc_kg_embs.append(doc_kg_emb)
        final_terms.append(list(term2vec.keys()))
    print("Embedding ended.")
    outputs = zip(final_terms, doc_kg_embs)
    return outputs
                   

def make_embedding(term2vec):
    if len(term2vec) == 0:
        return np.array([0]*512)
    avg = sum(list(term2vec.values())) / len(term2vec.values())  
    return avg


def test_env():
    train_text = {"text1":"Brexit (/ˈbrɛksɪt, ˈbrɛɡzɪt/;[1] a portmanteau of British and exit) is the withdrawal of the United Kingdom (UK) from the European Union (EU). Following a referendum held on 23 June 2016 in which 51.9 per cent of those voting supported leaving the EU, the Government invoked Article 50 of the Treaty on European Union, starting a two-year process which was due to conclude with the UK's exit on 29 March 2019 – a deadline which has since been extended to 31 October 2019.[2]",
                  "text2":"Withdrawal from the EU has been advocated by both left-wing and right-wing Eurosceptics, while pro-Europeanists, who also span the politica#l spectrum, have advocated continued membership and maintaining the customs union and single market. The UK joined the European Communities (EC) in 1973 under the Conservative government of Edward Heath, with continued membership endorsed by a referendum in 1975. In the 1970s and 1980s, withdrawal from the EC was advocated mainly by the political left, with the Labour Party's 1983 election manifesto advocating full withdrawal. From the 1990s, opposition to further European integration came mainly from the right, and divisions within the Conservative Party led to rebellion over the Maastricht Treaty in 1992. The growth of the UK Independence Party (UKIP) in the early 2010s and the influence of the cross-party People's Pledge campaign have been described as influential in bringing about a referendum. The Conservative Prime Minister, David Cameron, pledged during the campaign for the 2015 general election to hold a new referendum—a promise which he fulfilled in 2016 following pressure from the Eurosceptic wing of his party. Cameron, who had campaigned to remain, resigned after the result and was succeeded by Theresa May, his former Home Secretary. She called a snap general election less than a year later but lost her overall majority. Her minority government is supported in key votes by the Democratic Unionist Party.",
                  "text3":"The broad consensus among economists is that Brexit will likely reduce the UK's real per capita income in the medium term and long term, and that the Brexit referendum itself damaged the economy.[a] Studies on effects since the referendum show a reduction in GDP, trade and investment, as well as household losses from increased inflation. Brexit is likely to reduce immigration from European Economic Area (EEA) countries to the UK, and poses challenges for UK higher education and academic research. As of May 2019, the size of the divorce bill—the UK's inheritance of existing EU trade agreements—and relations with Ireland and other EU member states remains uncertain. The precise impact on the UK depends on whether the process will be a hard or soft Brexit."}

    x = naive_embedd(train_text.values())
    for z,y in x:
        print(z)
#test_env()