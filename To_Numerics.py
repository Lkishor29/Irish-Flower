
#Sklearn can work with Categorical as well
#But if someone wants to change them to Numerical
import pandas as pd
import numpy as np
data=pd.read_csv('Data.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x=np.array(x)
y=np.array(y)
i=0
y=y.reshape(1,150)
y_mod=np.zeros(150)
y_mod=y_mod.reshape(1,150)
for i in range(150):
    if(y[0][i]=='Iris-versicolor'):
        y_mod[0][i]=1
    elif(y[0][i]=='Iris-virginica'):
        y_mod[0][i]=2
        
        
y_mod=y_mod.reshape(150,1)
        
        
