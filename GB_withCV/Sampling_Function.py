#Magda Arnal
#24/05/2021
#Function to make cross validation on training and validation sets

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from imblearn.over_sampling import SMOTE


def Cross_Val_Groups(model, X, y, combination, n_splits = 10, balance = 'under'):

    
    f_score_train = []
    f_score_val = []
    
    # prec_train = []
    # prec_val = []
    
    # acc_train = []
    # acc_val = []
    
    # recall_train = []
    # recall_val = []
    
    roc_train = []
    roc_val = []
    
    if balance == 'under':
        rus = RandomUnderSampler(random_state=0)
    elif balance == 'over':
        sm = SMOTE(random_state=0)
    
    np.random.seed(0)
    strat = StratifiedKFold(n_splits = n_splits)
    
    for fold, (train_index, val_index) in enumerate(strat.split(X, y)):
        
        # print(train_index)
        # print(val_index)
        
        x_train = X[train_index]
        y_train = y[train_index]
        x_val = X[val_index]
        y_val = y[val_index]
        
        # unique, counts = np.unique(y_val, return_counts=True)
        # print('Counts before sampling:',np.asarray((unique, counts)).T)
        
        if balance == 'under':
            X_train, Y_train = rus.fit_resample(x_train, y_train)
            # unique, counts = np.unique(Y_train, return_counts=True)
            # print('Counts after undersampling:',np.asarray((unique, counts)).T)
            
            X_val, Y_val = rus.fit_resample(x_val, y_val)
            # unique, counts = np.unique(Y_val, return_counts=True)
            # print('Counts after undersampling:',np.asarray((unique, counts)).T)
            
        elif balance == 'over':
            #reduce the number of controls, because otherwise it takes forever
            casei = np.where(y_train==1)[0]
            cntrli  = np.where(y_train==0)[0]
            # print(len(casei))#529
            # print(len(cntrli))#54002
            #Subset 1:2 balancing
            ri = np.random.choice(cntrli, len(casei)*2, replace=False)
            #np.unique(ri).size
            idx = np.concatenate([casei,ri])
            #print(len(idx))#1545
            #Make the subset of feature predictors
            Xn=x_train[idx , : ]
            #print(Xn.shape)#(1545, 145)
            #Make the subset of the labels
            yn=y_train[idx]
            # unique, counts = np.unique(yn, return_counts=True)
            # print('Counts after oversampling:',np.asarray((unique, counts)).T)
            #Make the imputation with SMOTE
            X_train, Y_train = sm.fit_resample(Xn, yn)
            # unique, counts = np.unique(Y_train, return_counts=True)
            # print('Counts after oversampling:',np.asarray((unique, counts)).T)
            ######################################
            #Make the same for the validation test
            casei = np.where(y_val==1)[0]
            cntrli  = np.where(y_val==0)[0]
            #Subset 1:2 balancing
            ri = np.random.choice(cntrli, len(casei)*2, replace=False)
            idx = np.concatenate([casei,ri])
            Xn=x_val[idx , : ]
            yn=y_val[idx]
            X_val, Y_val = sm.fit_resample(Xn, yn)
            # unique, counts = np.unique(Y_val, return_counts=True)
            # print('Counts after oversampling:',np.asarray((unique, counts)).T)
        
        
        modTree = model.set_params(n_estimators=combination['n_estimators'],
                                   learning_rate=combination['learning_rate'],
                                   subsample= combination['subsample'], 
                                   max_depth=combination['max_depth'],
                                   loss=combination['loss'],
                                   random_state=0)
        modTree.fit(X_train, Y_train)        
        
        #Prediction on train fold
        pred_train = modTree.predict(X_train)
        # score = precision_recall_fscore_support(y_train, pred_train, average='binary')
        # prec_train.append(score[0])
        # recall_train.append(score[1])
        f_score_train.append(fbeta_score(Y_train, pred_train, average='binary', beta=1))        
        # acc = accuracy_score(y_train, pred_train)
        # acc_train.append(acc)
        #Calculate the roc curve
        mpred = modTree.predict_proba(X_train)
        pred=mpred[:,1]
        fpr, tpr, thresholds = roc_curve(Y_train, pred, pos_label=1)
        roc_train.append(auc(fpr, tpr))
        
        #Prediction on val fold 
        pred_val = modTree.predict(X_val)
        # score = precision_recall_fscore_support(y_val, pred_val, average='binary')
        # prec_val.append(score[0])
        # recall_val.append(score[1])
        f_score_val.append(fbeta_score(Y_val, pred_val, average='binary', beta=1))        
        # acc = accuracy_score(y_val, pred_val)
        # acc_val.append(acc)
        #Calculate the roc curve
        mpred = modTree.predict_proba(X_val)
        pred=mpred[:,1]
        fpr, tpr, thresholds = roc_curve(Y_val, pred, pos_label=1)
        roc_val.append(auc(fpr, tpr))
        
    # mean_prec_train = np.mean(np.array(prec_train))
    # std_prec_train = np.std(np.array(prec_train))
    # mean_prec_val = np.mean(np.array(prec_val))
    # std_prec_val = np.std(np.array(prec_val))
    # prec_metrics={'mean_prec_train': mean_prec_train, 
    #               'std_prec_train': std_prec_train,
    #               'mean_prec_val': mean_prec_val, 
    #               'std_prec_val': std_prec_val}
    
    # mean_recall_train = np.mean(np.array(recall_train))
    # std_recall_train = np.std(np.array(recall_train))
    # mean_recall_val = np.mean(np.array(recall_val))
    # std_recall_val = np.std(np.array(recall_val))
    # recall_metrics={'mean_recall_train': mean_recall_train, 
    #               'std_recall_train': std_recall_train,
    #               'mean_recall_val': mean_recall_val, 
    #               'std_recall_val': std_recall_val}
    
    mean_f_score_train = np.mean(np.array(f_score_train))
    std_f_score_train = np.std(np.array(f_score_train))
    mean_f_score_val = np.mean(np.array(f_score_val))
    std_f_score_val = np.std(np.array(f_score_val))
    fscore_metrics={'mean_fscore_train': mean_f_score_train, 
                  'std_fscore_train': std_f_score_train,
                  'mean_fscore_val': mean_f_score_val, 
                  'std_fscore_val': std_f_score_val}
    
    # mean_acc_train = np.mean(np.array(acc_train))
    # std_acc_train = np.std(np.array(acc_train))
    # mean_acc_val = np.mean(np.array(acc_val))
    # std_acc_val = np.std(np.array(acc_val))
    # acc_metrics={'mean_acc_train': mean_acc_train, 
    #               'std_acc_train': std_acc_train,
    #               'mean_acc_val': mean_acc_val, 
    #               'std_acc_val': std_acc_val}
    
    mean_roc_train = np.mean(np.array(roc_train))
    std_roc_train = np.std(np.array(roc_train))
    mean_roc_val = np.mean(np.array(roc_val))
    std_roc_val = np.std(np.array(roc_val))
    roc_metrics={'mean_roc_train': mean_roc_train, 
                  'std_roc_train': std_roc_train,
                  'mean_roc_val': mean_roc_val, 
                  'std_roc_val': std_roc_val}
    
    #return prec_metrics, recall_metrics, fscore_metrics, acc_metrics, roc_metrics
    return fscore_metrics,roc_metrics







    
    
    
   



