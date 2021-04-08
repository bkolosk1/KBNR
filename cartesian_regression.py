## Evaluation of the multitude of representations.
from utils import *

def get_unique_representation(rep_space):

    return [x.split("/")[-1].replace(".csv","") for x in rep_space if not "_ys" in x]


def construct_joint_representation(rep_subspace, path):

    subspaces = []
    for subspace in rep_subspace:
        subspace_tmp = np.loadtxt(path+subspace+".csv", delimiter = ",")
        subspaces.append(subspace_tmp)

    final_subspace = subspaces
    return final_subspace, rep_subspace

def find_best_learner(dataset, learner = "LR", optim_func = 'acc'):

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
    average = 'binary' if len(set(train_y)) == 2 else 'weighted'
    
    to_consider_combinations = []

    #for k in range(3,7):
    #    combs = itertools.combinations(unique_representations, k)
    #    for comb in combs:
    #        to_consider_combinations.append(comb)
    kg_types = ["complex", "transe", "rotate", "quate", "distmult", "simple"]
    kg_types = ["complex_entity", "transe_entity", "rotate_entity", "quate_entity", "distmult_entity", "simple_entity"]

    LM_types = ["distilbert-base-nli-mean-tokens","lsa","roberta-large-nli-stsb-mean-tokens","stat","xlm-r-large-en-ko-nli-ststb"]   
    merged = kg_types+LM_types
    to_consider_combinations = [merged]
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
            score = optim_func(dev_y, preds)
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
    f1_scr = f1_score(test_y, final_predictions, average = average)
    acc_scr = accuracy_score(test_y, final_predictions)
    prec_scr = precision_score(test_y, final_predictions,average = average)
    recall_scr = recall_score(test_y, final_predictions,average = average)
    return final_predictions, test_y, f1_scr, acc_scr, prec_scr, recall_scr, learner, final_combination

def store_out_object(out_object, path):
    with open(path, 'w') as fp:
        json.dump(out_object, fp)

if __name__ == "__main__":
    dataset_folder = "../../codename-fn/representations/*"
    all_problems = glob.glob(dataset_folder)
    


    for problem in all_problems:
        scorer = f1_score
        
        pname = problem.split("/")[-1]
        print(pname)
        if not pname == 'FakeNewsNet':
            continue
        if pname == 'pan2020' or pname=='LIAR_PANTS':
            scorer = accuracy_score
            
        final_predictions, test_labels, f1_scr, acc_scr, prec_scr, recall_scr, learner, final_comb = find_best_learner(problem, optim_func = scorer)
        
        out_obj = {"all_final_predictions":final_predictions.tolist(),
                   "final_f1":f1_scr,
                   'final_acc':acc_scr,
                   'final_prec':prec_scr,
                   'final_recall':recall_scr,
                   "learner":learner,
                   "combination":final_comb, 
                   "test_labels":test_labels.tolist()}
        store_out_object(out_obj, f"../results/{pname}_merged.json")
