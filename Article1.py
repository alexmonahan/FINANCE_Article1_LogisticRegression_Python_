#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 08:56:10 2017

@author: Alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alex Monahan
"""

import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("main_data.csv")

"""Get a sense of what the data looks like"""
print(df.columns) 

X = pd.DataFrame()
X['Investment_Grade'] = df['Below Investment Grade?']
X['GDP'] = df['GDP growth rate annual']
X['Unemployment'] = df['Unemployment Rate']
X['Inflation'] = df['Inflation Rate']
X['Interest'] = df['Interest rate']
X['Current_Account'] = df['Current account (surplus is 1, or deficit is zero)']
X['Oil_Exporter'] = df['Net oil exporter']
X['GovtDebt'] = df['Government Debt to GDP']

X = X.dropna(axis=0)
y = X['Investment_Grade']

X=X.drop(['Investment_Grade'], axis=1)

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=115)

def base_rate_model(X):
    y = [0]*len(X)
    return y
    
y_baserate = base_rate_model(X_test)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2', C=1)
model.fit(X_train, y_train)

print ("Base rate of every country is investment grade -- accuracy is %2.2f" % accuracy_score(y_test, y_baserate))
print ("Base rate accuracy is %2.2f" % accuracy_score(y_test, model.predict(X_test)))

logit = sm.Logit(y_train, X_train)
result = logit.fit()
print (result.summary())

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

print ("----BASE RATE MODEL-----")
base_roc_auc = roc_auc_score(y_test, base_rate_model(X_test))
print ("Base rate AUC is %2.2f" % base_roc_auc)
print ("" + classification_report(y_test, base_rate_model(X_test)))

print ("----LOGISTIC MODEL-----")
log_roc_auc = roc_auc_score(y_test, model.predict(X_test))
print ("Logistic AUC is %2.2f" % log_roc_auc)
print ("" + classification_report(y_test, model.predict(X_test)))

