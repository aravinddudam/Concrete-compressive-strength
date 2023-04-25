# importing packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# loding the data
df = pd.read_csv('Downloads/Concretedata.csv')

df.head()

# since name of the columns are huge converting them into short names
df.columns=['CC','BFS','FA','WC','SPC','CA','FAC','Age','CCS']

df.head()

# understanding the data
df.shape

## checking if any of the values are null
df.isnull().sum()

# there are 1030 instance and 9 features

df.info()

We can see that there all the 9 columns in the data are numerical

## understanding the 5 point summary of the data
df.describe()

## checking the relationship between the feature variables
df.corr()

from the above we can see that class variable CCS is not Strongly correlated with any other variable.

_

Since our problem statement is to find the if the Concrete compressive strength is suitable for a residental complex we will comverting the CCS columns into binary class

def new_class(x):
    if (x >= 17)&(x<=28):
        return(1)
    else:
        return(0)

df['Class'] = df['CCS'].apply(new_class)

import seaborn as sns

df['Class'].value_counts()

## creating different dataframes for dependent and independent variables
X = df.drop(['CCS','Class'],axis=1)
Y = df['Class']

## spliting the Data into Train Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import statsmodels.formula.api as smf

## fitting a Base model 

from sklearn.metrics import f1_score, make_scorer,precision_score

Precision = make_scorer(precision_score)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

LR = LogisticRegression()
DT = DecisionTreeClassifier()
Knn = KNeighborsClassifier(n_neighbors=4)
RF = RandomForestClassifier()
GBoost = GradientBoostingClassifier()
AB = AdaBoostClassifier()

models = []
models.append(('Logistic',LR))
models.append(('Decision_Tree',DT))
models.append(('KNN',Knn))
models.append(('RandomForest',RF))
models.append(('AdaBoost',AB))
models.append(('GradientBoost',GBoost))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(shuffle=True,n_splits=10,random_state=0)
    cv_results = model_selection.cross_val_score(model, X_train, y_train,cv=kfold, scoring= Precision )
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, np.mean(cv_results),np.std(cv_results,ddof=1)))
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

### Adaboost Model

## Standarding the Data
sc=StandardScaler()
X_std = sc.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_std,Y,test_size=0.3)

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(shuffle=True,n_splits=10,random_state=0)
    cv_results = model_selection.cross_val_score(model, X_train, y_train,cv=kfold, scoring= Precision )
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, np.mean(cv_results),np.std(cv_results,ddof=1)))
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

## Logistics Regression

##### Data Balancing

from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')

X_train_new, y_train_new = oversample.fit_resample(X_train, y_train.ravel())
class_balance = pd.Series(y_train_new).value_counts().plot.bar()
class_balance.set_title("Outcome ytrain")

## MODELING 3

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(shuffle=True,n_splits=10,random_state=0)
    cv_results = model_selection.cross_val_score(model, X_train_new, y_train_new,cv=kfold, scoring= Precision )
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, np.mean(cv_results),np.std(cv_results,ddof=1)))
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

## Logistic

# Comparing Between the models with hyperparameter tuning for each indiviual model

# Initialze the estimators
m0 = LogisticRegression()
m1 = DecisionTreeClassifier()
m2 = RandomForestClassifier()
m3 = KNeighborsClassifier()
m4 = GradientBoostingClassifier()
m5 = AdaBoostClassifier()
m6 = XGBClassifier()

Param0 =  { 'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'classifier__penalty': ['l1', 'l2', 'none'],
                      'classifier__C': [100, 10, 1.0, 0.1, 0.01],
                      'classifier__max_iter': [500000]
                     }

param1 = {}
param1['classifier__max_depth'] = [5,10,25,None]
param1['classifier__min_samples_split'] = [2,5,10]
param1['classifier'] = [m1]

param2 = {}
param2['classifier__n_estimators'] = [10, 50, 100]
param2['classifier__max_depth'] = [5, 10, 20]
param2['classifier'] = [m2]

param3 = {}
param3['classifier__n_neighbors'] = [1,2,3,4,5,6,7,8,9]
param3['classifier'] = [m3]             

param4 = {}
param4['classifier__n_estimators'] = [10, 50, 100]
param4['classifier__learning_rate'] =[0.1,0.05,0.02]
param4['classifier__max_depth'] = [5, 10]
param4['classifier'] = [m4]

param5 = {}
param5['classifier__n_estimators'] = [10, 50, 100, 250]
param5['classifier__learning_rate'] = [0.001, 0.01, 0.1, 1.0]
param5['classifier'] = [m5]

param6 = {
        'classifier__min_child_weight': [1, 5, 10],
        'classifier__gamma': [0.5, 1, 1.5, 2, 5],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__max_depth': [3, 4, 5],
        'classifier' : [m6]
        }

from sklearn.pipeline import Pipeline

pips = [Pipeline([('classifier', m0)]),Pipeline([('classifier', m1)]),Pipeline([('classifier', m2)]),Pipeline([('classifier', m3)]),Pipeline([('classifier', m4)]),
       Pipeline([('classifier', m5)])]
pars = [Param0 ,param1, param2, param3, param4, param5]

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

### using Precision as evaluation metric

print("starting Gridsearch")
for i in range(len(pars)):
    gs = GridSearchCV(pips[i], pars[i], cv=7 , n_jobs=-1, scoring = Precision )
    gs = gs.fit(X_train_new,y_train_new)
    print("finished Gridsearch")
    print(gs.best_score_)
    print(gs.best_params_)
    print("*"*125)

print("starting Gridsearch")
for i in range(len(pars)):
    gs = GridSearchCV(pips[i], pars[i], cv=7 , n_jobs=-1)
    gs = gs.fit(X_train_new,y_train_new)
    print("finished Gridsearch")
    print(gs.best_score_)
    print(gs.best_params_)
    print("*"*125)

print("starting Gridsearch")
for i in range(len(pars)):
    gs = GridSearchCV(pips[i], pars[i], cv=7 , n_jobs=-1, scoring = 'roc_auc' )
    gs = gs.fit(X_train_new,y_train_new)
    print("finished Gridsearch")
    print(gs.best_score_)
    print(gs.best_params_)
    print("*"*125)

#### Feature selection 

import statsmodels.api as sm

logit_model=sm.Logit(y_train_new,X_train_new)
result=logit_model.fit()
print(result.summary2())

The features with p-value less than 0.05 are considered to be the more relevant feature.

X.columns

From the above we can see that Features
Age
SPC
Cement

# Creating Model only with only the features 

e= pd.DataFrame(X_train_new)

e.columns = X.columns

X_selected = e[['CC','SPC','Age']]

X_selected .head()

len(X_selected)

len(y_train_new)

print("starting Gridsearch")
for i in range(len(pars)):
    gs = GridSearchCV(pips[i], pars[i], cv=7 , n_jobs=-1, scoring = Precision )
    gs = gs.fit(X_selected,y_train_new)
    print("finished Gridsearch")
    print(gs.best_score_)
    print(gs.best_params_)
    print("*"*125)

print("starting Gridsearch")
for i in range(len(pars)):
    gs = GridSearchCV(pips[i], pars[i], cv=7 , n_jobs=-1 )
    gs = gs.fit(X_selected,y_train_new)
    print("finished Gridsearch")
    print(gs.best_score_)
    print(gs.best_params_)
    print("*"*125)

print("starting Gridsearch")
for i in range(len(pars)):
    gs = GridSearchCV(pips[i], pars[i], cv=7 , n_jobs=-1, scoring = 'roc_auc' )
    gs = gs.fit(X_selected,y_train_new)
    print("finished Gridsearch")
    print(gs.best_score_)
    print(gs.best_params_)
    print("*"*125)
