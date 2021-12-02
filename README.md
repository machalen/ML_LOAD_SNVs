# Machine learning methods applied to genotyping data in late onset Alzheimer’s disease

Python scripts used in the article: *Machine learning methods applied to genotyping data capture interactions between single nucleotide variants in late onset Alzheimer’s disease*

Python 3.7.6 with Scikit-learn v0.22.1 module was used to build (train and test) the ML models. A train/test split was applied to have 80% of samples for training and 20% of samples for testing. Scripts are divided by the different ML methods used in the article:

- GB : GradientBoostingClassifier
- RF : RandomForestClassifier
- ET : ExtraTreesClassifier

#### Python scripts to run 10 fold cross-validation with the 80% of samples in the training split:

- \*\_DisGeNet_CV_Over.py : Oversampling to balance samples and AD SNVs as predictors
- \*\_DisGeNet_CV_Under.py : Undersampling to balance samples and AD SNVs as predictors
- \*\_refVars_CV_Over.py : Oversampling to balance samples and reference SNVs as predictors
- \*\_refVars_CV_Under.py : Undersampling to balance samples and reference SNVs as predictors

#### Python scripts to test the model with the 20% of samples in the testing split:

- \*\_finalModel.py : Final model with the definitive parameters selected in the cross-validation step

