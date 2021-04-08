# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:03:08 2021

@author: Bosec
"""


## Evaluation of the multitude of representations.
from utils import *
from sklearn.metrics import accuracy_score

def get_unique_representation(rep_space):

    return [x.split("/")[-1].replace(".csv","") for x in rep_space if not "_ys" in x]


def construct_joint_representation(rep_subspace, path):

    subspaces = []
    for subspace in rep_subspace:
        subspace_tmp = np.loadtxt(path+subspace+".csv", delimiter = ",")
        subspaces.append(subspace_tmp)

    final_subspace = subspaces
    return final_subspace, rep_subspace

def find_best_learner(dataset, learner = "LR"):

    train_path = dataset+"/train/"
    dev_path = dataset+"/dev/"
    test_path = dataset+"/test/"
          
    rep_files = glob.glob(train_path+"*")
    unique_representations = get_unique_representation(rep_files)
    targets = "_ys.csv"

    final_combination = None
    final_score = 0
    final_learner = None

    ## Target space
    train_y = np.loadtxt(train_path+targets).astype(int)
    dev_y = np.loadtxt(dev_path+targets).astype(int)
    test_y = np.loadtxt(test_path+targets).astype(int)

    print(set(train_y),set(dev_y),set(test_y))
    
    #to_consider_combinations = [('transe', 'distilbert-base-nli-mean-tokens', 'xlm-r-large-en-ko-nli-ststb')]
    LM_types = ["distilbert-base-nli-mean-tokens","lsa","roberta-large-nli-stsb-mean-tokens","stat","xlm-r-large-en-ko-nli-ststb"]
    to_consider_combinations = [LM_types]
    #for k in range(3,11):
    #    combs = itertools.combinations(unique_representations, k)
    #    for comb in combs:
    #        to_consider_combinations.append(comb)
  
    for combination in tqdm.tqdm(to_consider_combinations):
        
        representation_train,ss1 = construct_joint_representation(combination, train_path)
        representation_dev,ss2 = construct_joint_representation(combination, dev_path)
        enx = 0
        checked_reps_first = []
        checked_reps_second = []
        tmp_combination = []

        ## unit test rep shapes
        for a,b in zip(representation_train, representation_dev):
            if a.shape[1] != b.shape[1]:
                continue            
            checked_reps_first.append(a)
            checked_reps_second.append(b)
            tmp_combination.append(ss1[enx])
            enx += 1
            
        representation_train = np.hstack(checked_reps_first)
        representation_dev = np.hstack(checked_reps_second)
        
        assert representation_train.shape[1] == representation_dev.shape[1]
        
        for c in [1,10,100]:
            clf =  LogisticRegression(C = c, max_iter = 100000)
            clf.fit(representation_train, train_y)
            preds = clf.predict(representation_dev)
            score = accuracy_score(preds, dev_y)
            if score > final_score:
                logging.info(f"Found better combination of features: {combination} with score of {score}")
                final_learner = clf
                final_combination = tmp_combination
                final_score = score

    representation_test, ssf = construct_joint_representation(final_combination, test_path)
    #print(representation_test)
    #for el in representation_test:
    #    print(el.shape)
    representation_test = np.hstack(representation_test)
    final_predictions = final_learner.predict(representation_test)
    return final_predictions, test_y, accuracy_score(final_predictions, test_y), learner, final_combination

def store_out_object(out_object, path):
    with open(path, 'w') as fp:
        json.dump(out_object, fp)

if __name__ == "__main__":


    dataset_folder = "../../codename-fn/representations/*"
    all_problems = glob.glob(dataset_folder)
    
    for problem in all_problems:
        pname = problem.split("/")[-1]        
        if pname == 'ISOT':
          continue
        print(pname)         

        final_predictions, test_labels, score, learner, final_comb = find_best_learner(problem)
        out_obj = {"all_final_predictions":final_predictions.tolist(),"final_f1":score,"learner":learner,"combination":final_comb, "test_labels":test_labels.tolist()}
        store_out_object(out_obj, f"../results/{pname}_lm_only.json")
