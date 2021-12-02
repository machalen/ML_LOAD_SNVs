import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
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

###############################################################################
#############################Load and data in UKB##############################

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

Xtab=pd.DataFrame(pd.np.column_stack([labels, df]))
print(Xtab.shape)#(75738, 145)

con1=labels.iloc[:,1]
y=np.where(con1=='AD', 1, con1)
y=np.where(y=='Control', 0, y)
y=y.astype('int')

#Shuffle samples to avoid possible bias when sampling
Xtab, y = shuffle(Xtab, y)
#Check that the AD and controls are properly assigned after shuffle
con2=Xtab.iloc[:,1]
yt=np.where(con2=='AD', 1, con2)
yt=np.where(yt=='Control', 0, yt)
yt=yt.astype('int')
print(np.array_equal(y, yt))#TRUE!


#Get the training and test sets
X_trtab, X_tetab, y_train, y_test = train_test_split(Xtab, y, test_size=0.2)
print ('Train set:', X_trtab.shape,  y_train.shape)#Train set: (1388, 48) (1388, 3)
print ('Test set:', X_tetab.shape,  y_test.shape)#Test set: (596, 48) (596, 3)

#Check that the AD and controls are properly assigned after train_test split
con2=X_trtab.iloc[:,1]
ytr=np.where(con2=='AD', 1, con2)
ytr=np.where(ytr=='Control', 0, ytr)
ytr=ytr.astype('int')
print(np.array_equal(y_train, ytr))#TRUE!
con2=X_tetab.iloc[:,1]
yte=np.where(con2=='AD', 1, con2)
yte=np.where(yte=='Control', 0, yte)
yte=yte.astype('int')
print(np.array_equal(y_test, yte))#TRUE!

#Make undersampling for the train and test set
rus = RandomUnderSampler(random_state=0)

x_trtab, y_train = rus.fit_resample(X_trtab, y_train)
#Check that the AD and controls are properly assigned after train_test split
con2=x_trtab.iloc[:,1]
ytr=np.where(con2=='AD', 1, con2)
ytr=np.where(ytr=='Control', 0, ytr)
ytr=ytr.astype('int')
print(np.array_equal(y_train, ytr))#TRUE!
#Save the labels
x_trlab=x_trtab.iloc[:,0:3]
x_trlab.columns=labels.columns
x_trlab.to_csv(os.path.join(outDir, 'Training.eids.CV.txt'), index=None, sep='\t')
#Split and create the numeric np matrix
x_train=x_trtab.iloc[:,3:len(x_trtab.columns)]
a=x_trtab.iloc[:,147]
b=x_train.iloc[:,144]
a.compare(b)
x_train=x_train.to_numpy()

x_tetab, y_test = rus.fit_resample(X_tetab, y_test)
#Check that the AD and controls are properly assigned after train_test split
con2=x_tetab.iloc[:,1]
yte=np.where(con2=='AD', 1, con2)
yte=np.where(yte=='Control', 0, yte)
yte=yte.astype('int')
print(np.array_equal(y_test, yte))#TRUE!
#Save the labels
x_telab=x_tetab.iloc[:,0:3]
x_telab.columns=labels.columns
#Split and create the numeric np matrix
x_test=x_tetab.iloc[:,3:len(x_tetab.columns)]
a=x_tetab.iloc[:,147]
b=x_test.iloc[:,144]
a.compare(b)
x_test=x_test.to_numpy()

#Look at the number of AD and controls in train and test
unique, counts = np.unique(y_train, return_counts=True)
print('Counts in training:',np.asarray((unique, counts)).T)

unique, counts = np.unique(y_test, return_counts=True)
print('Counts in training:',np.asarray((unique, counts)).T)

#Define the model with parameters defined in the R script:
#1_SelectBestParameters_ET.R
#100|8|1|7
modTree = ExtraTreesClassifier(n_estimators=100,
                               min_samples_split=8,
                               min_samples_leaf= 1,
                               max_depth=7,
                               random_state=0)
                               
modTree.fit(x_train, y_train)

#Build the datagrame with the feature importances
PredictVars = pd.DataFrame(modTree.feature_importances_, columns=['FI'])
PredictVars['SNV']=df.columns
print(PredictVars.sort_values(by=['FI'], ascending=False))
PredictVars.to_csv(os.path.join(outDir, 'ET.DisGeNet.Predictors.CV.txt'), index=None, sep='\t')

#Calculate accuracy
yhat=modTree.predict(x_test)
print('Accuracy is:',metrics.accuracy_score(y_test, yhat))#Accuracy is: 0.7065217391304348

#Calculate ROC AUC
mpred = modTree.predict_proba(x_test)
pred=mpred[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
print('ROC AUC is:',metrics.auc(fpr, tpr))#ROC AUC is: 0.8196282293635792

#Calculate the fbeta score
print('f-beta is:', metrics.fbeta_score(y_test, yhat, average='binary', beta=1))#f-beta is: 0.7054545454545456

#Calculate PPV and NPV
c_m=metrics.confusion_matrix(y_test, yhat)
TN=c_m[0,0]
TP=c_m[1,1]
FN=c_m[1,0]
FP=c_m[0,1]

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print('Sensitivity is:',TPR)#Sensitivity is: 0.7028985507246377
print('Specificity is:',TNR)#Specificity is: 0.7101449275362319
print('Positive Predicted Value is:',PPV)#Positive Predicted Value is: 0.708029197080292
print('Negative Predicted Value is:',NPV)#Negative Predicted Value is: 0.7050359712230215

#Build the dataframe with the predictions
#PredictSamples = pd.DataFrame(y_test, columns=['y_test'])
x_telab['y_pred']=yhat
x_telab['Prob0']=mpred[:,0]
x_telab['Prob1']=mpred[:,1]
x_telab.to_csv(os.path.join(outDir, 'ET.DisGeNet.Samples.CV.txt'), index=None, sep='\t')

