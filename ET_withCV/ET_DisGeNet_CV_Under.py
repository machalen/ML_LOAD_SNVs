#Magda Arnal
#19/05/2021

import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.utils import shuffle
from Sampling_Function import Cross_Val_Groups
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
#import getopt
from optparse import OptionParser

#Disable warnings: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

###############################################################################
###############################################################################
parser = OptionParser()

parser.add_option('-m', '--inputMtrx', help='Input numeric Matrix in .txt format where columns are predictors and rows are samples', 
                  metavar='FullPath/Mtrx.txt')
parser.add_option('-l', '--inputLabels',
                  help='Labels corresponding to the rows in the input matrix in .txt format',
                  metavar='FullPath/labels.txt')
parser.add_option('-o', '--outputDir',
                  help='Output path to the results',
                  metavar='FullPath')

(options, args) = parser.parse_args()

################################################################################
#Assess the input variables
#Convert <class 'optparse.Values'> to dictionary
option_dict = vars(options)
#Save elements in the dictionary to variables
inMtrx = option_dict['inputMtrx']
inLab = option_dict['inputLabels']
outDir = option_dict['outputDir']

#########################################################################
###############Load and data in UKB######################################

df = pd.read_csv(inMtrx, sep="\t")
#df.head()
print(df.shape)#(75738, 145)

labels = pd.read_csv(inLab, sep="\t")
#labels.head()
print(labels.shape)#(75738, 3)

#########################################################################
######################PreProcess UKB data################################
#Import libraries used here

#Select rows for case and controls
rows = labels.Cond.values
casei = np.where(rows=='AD')[0]
cntrli  = np.where(rows=='Control')[0]
print(len(casei))#738
print(len(cntrli))#75000

###############################################################################
#####################Get the best parameters in ET#############################
np.random.seed(21)

#Use all the dataset as input and balance in the function
#Make the subset of feature predictors
X=df.to_numpy()
print(X.shape)#(75738, 145)
#Get the training and test sets
con1=labels.iloc[:,1]
y=np.where(con1=='AD', 1, con1)
y=np.where(y=='Control', 0, y)
y=y.astype('int')
X, y = shuffle(X, y, random_state=1)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1)
print ('Train set:', X_train.shape,  y_train.shape)#Train set: (1388, 48) (1388, 3)
print ('Test set:', X_test.shape,  y_test.shape)#Test set: (596, 48) (596, 3)

# define the model with default hyperparameters
model=ExtraTreesClassifier()

# define the grid of values to search
grid = dict()
grid['n_estimators'] = [50, 60, 70, 80, 100]
grid['min_samples_split'] = [2, 5, 8]
grid['min_samples_leaf'] = [1, 2, 5]
grid['max_depth'] = [3, 7, 9, None]

# grid = dict()
# grid['n_estimators'] = [50, 60]
# grid['min_samples_split'] = [2, 5]
# grid['min_samples_leaf'] = [5]
# grid['max_depth'] = [9, None]

# unique, counts = np.unique(y_test, return_counts=True)
# print('Counts after train test:',np.asarray((unique, counts)).T)

# unique, counts = np.unique(y_train, return_counts=True)
# print('Counts after train test:',np.asarray((unique, counts)).T)

#Test all the combinations of hyperparameters
metrics_fscore_Under = {}
metrics_roc_Under = {}

keys_list = list(grid)
now1 = datetime.now()
np.random.seed(1)

for e in range(len(grid[keys_list[0]])):
    ne=grid[keys_list[0]][e]
    for l in range(len(grid[keys_list[1]])):
        ms=grid[keys_list[1]][l]
        for s in range(len(grid[keys_list[2]])):
            ml=grid[keys_list[2]][s]
            for m in range(len(grid[keys_list[3]])):
                md=grid[keys_list[3]][m]
                #print the combinations
                print(ne,ms,ml,md)
                c=(ne,ms,ml,md)
                combination = {'n_estimators':ne,
                               'min_samples_split':ms,
                               'min_samples_leaf':ml,
                               'max_depth': md}
                metrics_fscore_Under[c], metrics_roc_Under[c] = Cross_Val_Groups(model, X_train, y_train, combination, n_splits = 10, balance = 'under')

now2 = datetime.now()
print(now2-now1)

##################################################################################################
##################################################################################################
#Try to get the best metrics for each combination of parameters in the
#dictionary of dictionaries

keys_list = list(metrics_fscore_Under)
fscore_train_mean = []
fscore_train_std = []
fscore_val_mean = []
fscore_val_std = []
hyperparam = []
for i in range(len(keys_list)):
    fscore_val_mean.append(metrics_fscore_Under[keys_list[i]]['mean_fscore_val'])
    fscore_val_std.append(metrics_fscore_Under[keys_list[i]]['std_fscore_val'])
    fscore_train_mean.append(metrics_fscore_Under[keys_list[i]]['mean_fscore_train'])
    fscore_train_std.append(metrics_fscore_Under[keys_list[i]]['std_fscore_train'])
    hyperparam.append('|'.join(str(p) for p in keys_list[i]))

#Save a table with the results
fscore_df = pd.DataFrame()
fscore_df['HyperParam']  = hyperparam
fscore_df['ValMean']  = fscore_val_mean
fscore_df['ValStd']  = fscore_val_std
fscore_df['TrainMean']  = fscore_train_mean
fscore_df['TrainStd']  = fscore_train_std
fscore_df.to_csv(os.path.join(outDir, 'ET.DisGeNet.fscore.Under.CV.txt'), index=None, sep='\t')

#Check the best metrics
max_value = max(fscore_val_mean)
max_index = fscore_val_mean.index(max_value)
#keys_list[max_index]
print('Best fscore mean {} is with {}'.format(max_value.round(4), keys_list[max_index]))

min_value = min(fscore_val_std)
min_index = fscore_val_std.index(min_value)
#keys_list[min_index]
print('Best fscore std {} is with {}'.format(min_value.round(4), keys_list[min_index]))


######################################
keys_list = list(metrics_roc_Under)
roc_val_mean = []
roc_val_std = []
roc_train_mean = []
roc_train_std=[]
hyperparam = []
for i in range(len(keys_list)):
    roc_val_mean.append(metrics_roc_Under[keys_list[i]]['mean_roc_val'])
    roc_val_std.append(metrics_roc_Under[keys_list[i]]['std_roc_val'])
    roc_train_mean.append(metrics_roc_Under[keys_list[i]]['mean_roc_train'])
    roc_train_std.append(metrics_roc_Under[keys_list[i]]['std_roc_train'])
    hyperparam.append('|'.join(str(p) for p in keys_list[i]))


#Save a table with the results
roc_df = pd.DataFrame()
roc_df['HyperParam']  = hyperparam
roc_df['ValMean']  = roc_val_mean
roc_df['ValStd']  = roc_val_std
roc_df['TrainMean']  = roc_train_mean
roc_df['TrainStd']  = roc_train_std
roc_df.to_csv(os.path.join(outDir, 'ET.DisGeNet.roc.Under.CV.txt'), index=None, sep='\t')

#Check the best metrics
max_value = max(roc_val_mean)
max_index = roc_val_mean.index(max_value)
#keys_list[max_index]
print('Best ROC AUC mean {} is with {}'.format(max_value.round(4), keys_list[max_index]))

min_value = min(roc_val_std)
min_index = roc_val_std.index(min_value)
#keys_list[min_index]
print('Best ROC AUC std {} is with {}'.format(min_value.round(4), keys_list[min_index]))



