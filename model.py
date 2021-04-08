3# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:08:59 2020

@author: Bosec
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns

import matplotlib.pyplot as plt

import model_helper 

def predict(test_dataset, net, batch_size=32):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    orgs, preds = evaluate_env(net, test_loader, mode="TEST", print_ = True)
   
    return orgs, preds
    

def prepare_loaders(train_dataset, val_dataset, batch_size=20):
    #torch.manual_seed(1903)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader 


def evaluate_env(net, test_loader, device=torch.device("cuda"), mode = "VALID", print_ = False):
    orgs = []
    preds = []
    net.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels =  inputs.to(device), labels.to(device)
            outputs = net(inputs.float())
            val_loss = val_loss + criterion(outputs, labels)
            logits = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            guessed = np.argmax(logits,axis=1)
            orgs = orgs + labels.tolist()
            preds = preds + guessed.tolist()
   # print()
    
    if print_:
        print(f'{mode} accuracy: {accuracy_score(orgs, preds)}')
        print(f'{mode} F1-score: {f1_score(orgs, preds)}')
        print(f'{mode} precision: {precision_score(orgs, preds)}')
        print(f'{mode} recall: {recall_score(orgs, preds)}')
    return orgs, preds, val_loss

def predict(test_dataset, net, batch_size=32):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    evaluate_env(net, test_loader, mode="TEST", print_ = True)
    
    
def fit(model, test_loader, device=torch.device("cuda")):
    net = model_helper.FiveNet(2576 , p = 0.7)
    net.load_state_dict(model)
    net = net.to(device)
    net.eval()
    preds = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels =  inputs.to(device), labels.to(device)
            outputs = net(inputs.float())
            
            logits = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            guessed = np.argmax(logits,axis=1)
            preds = preds + guessed.tolist()
    return preds        
        
    
def train_NN(train_dataset, val_dataset, dims = 1552, e_max = 8, max_epochs = 100, batch_size = 300, lr = 0.005, dropout = 0.5,dataset = "pan2020"):
    #plt.clf()
    #plt.figure()
 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, val_loader = prepare_loaders(train_dataset, val_dataset, batch_size)

    net = model_helper.SqrtNet(dims, e_max = e_max, p = dropout)
    print(net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)                                          
    
    results = []
    scores = {}
    min_los_val = 9999999999999
    no_improv = 0
    max_no_improv = 25
    scores_plt = {  "validation" : [], "train" : [] }
    for epoch in tqdm(range(max_epochs), total = max_epochs):  
        #running_loss = 0.0
        if no_improv == max_no_improv:
            print()
            print('EARLY STOPPING!')
            break
        
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels =  inputs.to(device), labels.to(device)
            optimizer.zero_grad()
    
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        o, p, loss_train = evaluate_env(net, train_loader, device, mode="TRAIN")
        
        score = accuracy_score(o, p)
        scores_plt["train"].append(score)
            
        o, p, loss_val = evaluate_env(net, val_loader, device, mode="VALID")
        score = accuracy_score(o, p)
        scores_plt["validation"].append(score)
        
        
        scores[score] = net
        results.append(score)
        if min_los_val > loss_val:
            min_los_val = loss_val
            no_improv = 0
        else:
            no_improv = no_improv + 1
        
    scores_plt["epochs"]  = list(range(len(scores_plt['validation'])))

    df = pd.DataFrame(scores_plt)
    name = dataset+"_"+str(lr)+"_dpot_"+str(dropout)+".pkl"
    import pickle
    with open("log_data/"+name, 'wb') as f:
        pickle.dump(df, f)

    #plot.figure.savefig(,dpi=300)

    return scores[max(list(scores.keys()))]
