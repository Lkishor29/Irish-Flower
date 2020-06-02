import pandas as pd
data=pd.read_csv('Data.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.svm import SVC
model=SVC(kernel='poly',degree=4)
model.fit(x_train,y_train)

print(model.score(x_test,y_test))
import numpy as np
test=[4,3,5,1]
test=np.array(test).reshape(1,4)
y_pred=model.predict(test)
print(y_pred)