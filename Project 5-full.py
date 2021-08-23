# Author: Nima Farnoodian <nima.farnoodian@student.uclouvain.be>, Louvain-La-Neuve, Belgium.

import pickle
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import pandas as pd
from scipy import stats
import time
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from mlxtend.evaluate import bootstrap_point632_score 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC 


# Loading Data
infile = open('train1.pickle','rb')
train_1 = pickle.load(infile)
infile.close()

infile = open('train2.pickle','rb')
train_2 = pickle.load(infile)
infile.close()

infile = open('test.pickle','rb')
test = pickle.load(infile)
infile.close()

# Concatinating the training datasets
train=pd.concat([train_1[0], train_2[0]], axis=0)
y_train=np.concatenate((train_1[1],train_2[1]))
train['label']=y_train
del(train_1)
del(train_2)
del(y_train)

print('Train:')
print('\tThe number of missing Values:',train.isnull().sum().sum())
print('\tThe average number of missing Values for each attribute:',np.mean(train.isnull().sum()))
print('\tThe standard deviation of the average number of missing Values for each attribute:',np.std(train.isnull().sum()))

print('Test:')
print('\tThe number of missing Values:',test.isnull().sum().sum())
print('\tThe average number of missing Values for each attribute:',np.mean(test.isnull().sum()))
print('\tThe standard deviation of the average number of missing Values for each attribute:',np.std(test.isnull().sum()))

# finding mean and mode for imputation
imput_values=[]
for i in range(1306624):
    val=np.mean(train.iloc[:,i])
    imput_values.append(val)
for i in range(1306624,1306630):
    val=st.mode(train.iloc[:,i])
    imput_values.append(val)

    # imputing
for i in range (len(imput_values)):
    train.iloc[:,i].fillna(imput_values[i],inplace=True)
    test.iloc[:,i].fillna(imput_values[i],inplace=True)

print('Train:')
print('\tThe number of missing Values:',train.isnull().sum().sum())
print('\tThe average number of missing Values for each attribute:',np.mean(train.isnull().sum()))
print('\tThe standard deviation of the average number of missing Values for each attribute:',np.std(train.isnull().sum()))

print('Test:')
print('\tThe number of missing Values:',test.isnull().sum().sum())
print('\tThe average number of missing Values for each attribute:',np.mean(test.isnull().sum()))
print('\tThe standard deviation of the average number of missing Values for each attribute:',np.std(test.isnull().sum()))

#1-Preprocessing
    #1-1. Removing Constant and Quasi-constant features

counter=0
feature_constant=[]
feature_non_constant=[]
for i in range (1306624):
    var=np.var(train.iloc[:,i])
    if np.round(var,3)<=0.01:
        if np.round(var,3)==0:
            counter+=1
        feature_constant.append(i)
    else:
        feature_non_constant.append(i)


print('There are', counter, 'constant features')
print('There are', len(feature_constant), 'Quasi constant features')
none_image_features=[i for i in range(1306624,1306630) ]
print('There are', len(feature_non_constant)+len(none_image_features), 'features') 
features=feature_non_constant+none_image_features
features.append('label')
# Selecting non-constant features
train=train[features]
test=test[features[:-1]]

    #1-2. Normalizing Data (Image related attributes)
feature_for_normalization=train.iloc[1:5,:-7].columns
for feat in feature_for_normalization:
    mean=np.mean(train[feat])
    std=np.std(train[feat])
    train[feat]= (train[feat]-mean)/std
    test[feat]= (test[feat]-mean)/std
    #1-3. Treating Categorical values
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train[none_image_features])
one_hot_train=enc.transform(train[none_image_features])
one_hot_test=enc.transform(test[none_image_features])
train_categorical=pd.DataFrame(one_hot_train.toarray(),columns=[i for i in range(64)])
test_categorical=pd.DataFrame(one_hot_test.toarray(),columns=[i for i in range(64)])
train=pd.concat((train[feature_for_normalization],train_categorical,train['label'])) 


X_train, X_val, y_train, y_val = train_test_split(train.iloc[:,:-1], train.iloc[:,-1], test_size=0.2, random_state=1)


# Finding the most relevant features using Linear SVM
lsvc = LinearSVC(C=.01, penalty="l1", dual=False ,max_iter=2000).fit(X_train, y_train)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X_train)
X_new_val = model.transform(X_val)
X_new_test=model.transform(test)

# Find the best fit for MLP using Grid Search
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=100)
parameter_space = {
    'hidden_layer_sizes': [(10,20,4),(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.05 ],
    'learning_rate': ['constant','adaptive']
}

clf = GridSearchCV(mlp, parameter_space,scoring='balanced_accuracy', n_jobs=-1, cv=3)
clf.fit(X_new,y_train)
best_mlp=clf.best_estimator_

# Find the best fit for SVC using Grid Search

# defining parameter range 
svc = SVC()
param_grid = {'C': [0.01,0.1, 1, 10, 12,15,50,100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001,0.00001],
              'kernel': ['rbf','poly','sigmoid'],
              'degree': [2,3,4]} 
  
grid = GridSearchCV(svc, param_grid, refit = True, verbose = 3, scoring='balanced_accuracy')
  
grid.fit(X_new,y_train)     # fitting the model for grid search
best_svc=grid.best_estimator_

# Find the best fit for Logistig Regression using Grid Search
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,scoring='balanced_accuracy',verbose=2)
logreg_cv.fit(X_new,y_train)
best_lg=logreg_cv.best_estimator_

# Define a stacked model with the best classifiers found (MLP, SVC, and LG) and run CrossValidation using Stratified K fold
estimators=[
    ('mlp', best_mlp), 
    ('svc', best_svc),
    ('lg',best_lg)]

stacked_clf = StackingClassifier(
    estimators=estimators,
    stack_method="predict",
    n_jobs=-1
)
 
# Check the model performance
cv = StratifiedKFold(n_splits=5, random_state=1) # StratifiedKFold ensures that the sampling will consider the unbalancedness
scores = cross_val_score(stacked_clf, X_new, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

stacked_clf.fit(X_new, y_train)
test_pred=stacked_clf.predict(X_new_val)
print('BCR Test score:',balanced_accuracy_score(test_pred, y_val) )

# Evaluating the model using bootstrap 0.632+ technique. This model evaluation is more robust to overfitting. Here the expected BCR is computed.
# Model accuracy
scores = bootstrap_point632_score(stacked_clf, X_new, y_train.to_numpy(), method='.632+',scoring_func =balanced_accuracy_score)
acc = np.mean(scores)
print('BCR: %.2f%%' % (100*acc))


# Confidence interval
lower = np.percentile(scores, 2.5)
upper = np.percentile(scores, 97.5)
print('95%% Confidence interval: [%.2f, %.2f]' % (100*lower, 100*upper))
print('Expected BCR:', acc) # Computing expected BCR