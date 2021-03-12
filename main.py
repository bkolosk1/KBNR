import san
from src_end2end import statistical_features
import lsa_features
import umap_features
import pickle
import numpy as np
import pandas as pd
import skopt
from skopt import gp_minimize
from sklearn import preprocessing
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.metrics import f1_score
import argparse
import os
from sentence_transformers import SentenceTransformer

st_models = ["roberta-large-nli-stsb-mean-tokens", "xlm-r-large-en-ko-nli-ststb", "distilbert-base-nli-mean-tokens"]
dataset = None
texts = None
ys = None

def embedd_bert(text, st_model = 'paraphrase-distilroberta-base-v1', split = 'train'):    
    paths = "temp_berts/"+st_model+"_"+split+'.pkl'
    if os.path.isfile(paths):
        sentence_embeddings = pickle.load(open(paths,'rb')) 
        return sentence_embeddings
    model = SentenceTransformer(st_model)
    sentence_embeddings = model.encode(text)
    with open(paths, 'wb') as f:
        pickle.dump(sentence_embeddings, f)
    return sentence_embeddings


def prep_kgs(kg_emb, split='train'):
    embs = []
    global dataset
    path_in = "kg_emb_dump/"+dataset+"/"+split+"_"+kg_emb+'.pkl'
    with open(path_in, "rb") as f:
        kgs_p = pickle.load(f)
    for x,y in kgs_p:
        embs.append(y)
    return embs

def prep_all_feats(kg_emb, lsa_feats,  lsa_dims, umap_feats, umap_dims, umap_neigh, umap_min_dist, split = "train"):
    global texts
    LM_feats_train = prep_features_textual(texts[split], lsa_feats, lsa_dims, umap_feats, umap_dims, umap_neigh, umap_min_dist, split)
    kg_feats_train = prep_kgs(kg_emb, split)
    final_feats = np.hstack((LM_feats_train, kg_feats_train))
    return preprocessing.scale(final_feats)


def prep_features_textual(texts, lsa_feats, lsa_dims, umap_feats, umap_dims, umap_neigh, umap_min_dist, split):   
    staticstical = statistical_features.fit_space(texts)
    lsa = lsa_features.fit_space(texts, lsa_feats, lsa_dims, split)
    umap = umap_features.fit_space(texts, umap_feats, umap_dims, umap_neigh, umap_min_dist, split)
    bertz = embedd_bert(texts, st_models[0]) 
    bertz2 = embedd_bert(texts, st_models[1]) 
    bertz3 = embedd_bert(texts, st_models[2]) 
    s_b = np.hstack((staticstical, bertz))
    s_b = np.hstack((s_b, bertz2))
    s_b = np.hstack((s_b, bertz3))
    s_b = np.hstack((s_b, lsa))
    s_b = np.hstack((s_b,umap))
    return s_b



#SAN
num_epochs = None
num_heads = None
batch_size = None
dropout = None
hidden_lyr_size = None
learning_rate = None    
#LSA
lsa_n_features = None
lsa_n_dimensions = None
#KG
kg_id_mask = None
logs_file_ptr = None
#UMAP
umap_n_features = None
umap_n_dimensions = None
umap_neigh = None
umap_min_dist = None





#PARAMZ
bo_num_epochs = Integer(low=10, high=1000, name='bo_num_epochs')
bo_num_heads = Integer(low=1, high=4, name='bo_num_heads')
bo_learning_rate = Real(low=1e-4, high=1e-1, prior='log-uniform', name='bo_learning_rate')
bo_hidden_lyr_size = Real(low = 512, high =8192, name = 'bo_hidden_lyr_size')#Categorical([512,1024,2048,4096,8192 ], name = 'bo_hidden_lyr_size')
bo_batch_size = Categorical([32, 64, 128, 256, 512], name='bo_batch_size')
bo_dropout = Real(low=0.1, high=0.9, prior='uniform', name='bo_dropout')
#
bo_lsa_n_features = Categorical([1250, 2500, 5000, 10000, 15000, 20000], name='bo_lsa_n_features')
bo_lsa_n_dimensions = Categorical([128, 256, 512,1024], name='bo_lsa_n_dimensions')
#UMAP
bo_umap_n_features = Categorical([1250, 2500, 5000, 10000, 15000, 20000], name='bo_umap_n_features')
bo_umap_n_dimensions = Integer(low=5, high=100, name='bo_umap_n_dimensions')
bo_umap_neigh = Integer(low = 2, high = 5, name='bo_umap_neigh')
bo_umap_min_dist = Real(low=0.1, high=0.9,  prior='uniform', name='bo_umap_min_dist')

bo_kg_type = Categorical(["transe", "quate", "simple", "rotate", "distmult", "complex"], name = 'bo_kg_type')# "distmult", "complex", "simple", "rotate", "quate"], name = 'bo_kg_type')
import logging

dimensions = [bo_lsa_n_features, bo_lsa_n_dimensions, 
              bo_umap_n_features, bo_umap_n_dimensions, bo_umap_neigh, bo_umap_min_dist,
              bo_num_epochs, bo_num_heads, bo_hidden_lyr_size, bo_batch_size, bo_learning_rate, bo_dropout, bo_kg_type]




@use_named_args(dimensions=dimensions)
def fitness(bo_lsa_n_features, bo_lsa_n_dimensions, 
              bo_umap_n_features, bo_umap_n_dimensions, bo_umap_neigh, bo_umap_min_dist,
              bo_num_epochs, bo_num_heads, bo_hidden_lyr_size, bo_batch_size, bo_learning_rate, bo_dropout, bo_kg_type):

    global iteration, num_steps, lstm_size, init_epoch, max_epoch, learning_rate_decay, dropout_rate, \
    umap_n_features, umap_n_dimensions, umap_neigh, umap_min_dist, \
    init_learning_rate, batch_size, kd_id_mask, ys, logs_file_ptr
        
    #UMAP
    umap_n_features = int(bo_umap_n_features)
    umap_n_dimensions = int(bo_umap_n_dimensions)
    umap_neigh = int(bo_umap_neigh)
    umap_min_dist = float(bo_umap_min_dist)
    #LSA
    lsa_n_features = int(bo_lsa_n_features)
    lsa_n_dimensions = int(bo_lsa_n_dimensions)
    #SAN
    num_epochs = int(bo_num_epochs)
    num_heads = int(bo_num_heads)
    batch_size = int(bo_batch_size)
    dropout = float(bo_dropout)
    hidden_lyr_size = int(bo_num_epochs)
    learning_rate = float(bo_learning_rate)    
    #KG
    kg_id_mask = bo_kg_type    
    x_train = prep_all_feats(kg_id_mask, lsa_n_features, lsa_n_dimensions, umap_n_features, umap_n_dimensions, umap_neigh, umap_min_dist, split = 'train') 
    
    clf = san.SAN(num_epochs = num_epochs, num_heads = num_heads, batch_size = batch_size, dropout = dropout, learning_rate = learning_rate, hidden_layer_size = hidden_lyr_size)
    #PREP FEATS
    logging.info('Preprearing train features')
    #x_train = prep_all_feats(kg_id_mask, lsa_n_features, lsa_n_dimensions, 'train') 
    logging.info('Finished train features')


    clf.fit(x_train, ys["train"])
    del x_train
    
    logging.info('Preprearing dev features')
    x_dev = prep_all_feats(kg_id_mask, lsa_n_features, lsa_n_dimensions, umap_n_features, umap_n_dimensions, umap_neigh, umap_min_dist,  split = 'dev') 
    predict_dev = clf.predict(x_dev)
    del x_dev
   
    logging.info('Finished dev features')

    dev_ys = ys["dev"]    
    dev_error = 1 - f1_score(dev_ys, predict_dev, average='weighted')
    print("DEV F1-score", 1 - dev_error)
    logging.info('DEV F1-score: ' + str(1-dev_error))
    
    log_line = [dev_error, bo_lsa_n_features, bo_lsa_n_dimensions, bo_umap_n_features, bo_umap_n_dimensions, bo_umap_neigh, bo_umap_min_dist, bo_num_epochs, bo_num_heads, bo_hidden_lyr_size, bo_batch_size, bo_learning_rate, bo_dropout, bo_kg_type]
    log_line_s = '\t'.join(log_line)
    logs_file_ptr.write(log_line_s+"\n")
    do_test = False
    if do_test:
        logging.info('Preprearing test features')
        x_test = prep_all_feats(kg_id_mask, lsa_n_features, lsa_n_dimensions, umap_n_features, umap_n_dimensions, umap_neigh, umap_min_dist, split = 'test') 
        logging.info('Finished test features')

        predict_test = clf.predict(x_test)
        test_err = 1 - f1_score(ys["test"], predict_test, average='weighted')
        logging.info('TEST F1-score: ' + str(1-test_err))

        print("TEST  F1-score", 1 - test_err)
    
    return dev_error

umap_n_features = None
umap_n_dimensions = None
umap_neigh = None
umap_min_dist = None



if __name__ == "__main__":
    #CLEAN UP
    folders = ["temp_berts/","umap_pickles/","lsa_pickles/"]
    for folder in folders:
        for f in os.listdir(folder):
            os.remove(folder+f)
    
    my_parser = argparse.ArgumentParser

    # Preparing the arguments
    parser = argparse.ArgumentParser(description='Execute the binderz')
    parser.add_argument("--dataset", help="Dataset to evaluate on.", dest = 'dataset', default='pan2020', required=True)
    parser.add_argument("--n_calls", help="n_calls for bayes to evaluate on.", dest = 'n_calls', default=11, required=True)
    parser.add_argument("--lsa_n_features", help="lsa_n_features to evaluate on.", dest = 'lsa_n_features', default=2500)
    parser.add_argument("--lsa_n_dimensions", help="lsa_n_dimensions to evaluate on.", dest = 'lsa_n_dimensions', default=512)
    
    parser.add_argument("--umap_n_features", help="umap_n_features to evaluate on.", dest = 'umap_n_features', default=2500)
    parser.add_argument("--umap_n_dimensions", help="umap_n_dimensions to evaluate on.", dest = 'umap_n_dimensions', default=64)
    parser.add_argument("--umap_neigh", help="umap_neigh to evaluate on.", dest = 'umap_neigh', default=5)
    parser.add_argument("--umap_min_dist", help="umap_min_dist to evaluate on.", dest = 'umap_min_dist', default=0.1)


    parser.add_argument("--num_epochs", help="num_epochs to evaluate on.", dest = 'num_epochs', default=722)
    parser.add_argument("--num_heads", help="num_heads to evaluate on.", dest = 'num_heads', default=2)
    parser.add_argument("--batch_size", help="batch_size to evaluate on.", dest = 'batch_size', default=256)
    parser.add_argument("--dropout", help="dropout to evaluate on.", dest = 'dropout', default=0.3948)
    parser.add_argument("--hidden_lyr_size", help="hidden_lyr_size to evaluate on.", dest = 'hidden_lyr_size', default=2048)
    parser.add_argument("--rand_seed", help="rand_seed to evaluate on.", dest = 'rand_seed', default=42)

    parser.add_argument("--learning_rate", help="learning_rate to evaluate on.", dest = 'learning_rate', default=0.0001)
    parser.add_argument("--kg_id_mask", help="Dataset to evaluate on.", dest = 'kg_id_mask', default='transe')





    args = parser.parse_args()
    print(args)
    #exit()
    dataset = str(args.dataset)
    logs_file = "logs/outs_"+args.dataset+".log"
    logs_file_ptr = open(logs_file, 'w')

#bo_lsa_n_features, bo_lsa_n_dimensions, 
              #bo_umap_n_features, bo_umap_n_dimensions, bo_umap_neigh, bo_umap_min_dist,
             # bo_num_epochs, bo_num_heads, bo_hidden_lyr_size, bo_batch_size, bo_learning_rate, bo_dropout, bo_kg_type)
    
    default_parameters = [int(args.lsa_n_features), int(args.lsa_n_dimensions), 
                          int(args.umap_n_features), int(args.umap_n_dimensions), int(args.umap_neigh), float(args.umap_min_dist),
                          int(args.num_epochs), int(args.num_heads), int(args.hidden_lyr_size), int(args.batch_size), float(args.learning_rate), float(args.dropout),
                          str(args.kg_id_mask)]


   # global texts, ys

    texts = {}
    ys = {}
    for thing in ["train", "dev", "test"]:
        path_in = "data/final/"+dataset+"/"+thing+'.csv'
        df = pd.read_csv(path_in, encoding='utf-8')
        texts[thing] = df.text_a.to_list()
        ys[thing] = df.label.to_list()
    
    search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI',
                                n_calls=int(args.n_calls),
                                n_jobs = -1,
                                x0=default_parameters,
                                random_state=int(args.rand_seed))
    with open("bayes_"+dataset+".opt", "wb") as f:
       pickle.dump(search_result, f) 
    
    
    logs_file_ptr.close()



