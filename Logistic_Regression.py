#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:57:53 2020

@author: harmeet
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv('Data.csv')
x=pd.DataFrame(data)
x=x.to_numpy()
np.random.shuffle(x)
b=x[:,-1]
a=x[:,0:4]
a=a.astype('float32')
x_train,x_test,y_train,y_test=train_test_split(a,b,test_size=0.2)
tot_classes=len(np.unique(y_train))
num_classes=np.unique(y_train)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression(solver='liblinear',multi_class='ovr')
model.fit(x_train,y_train)
pred=model.predict(x_test)
from sklearn.metrics import classification_report
print(accuracy_score(y_test,pred))
print(classification_report(y_test, pred))