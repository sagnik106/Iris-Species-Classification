import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

d = pd.read_csv("Iris.csv")
x=d.iloc[:,1:-1].values
y=d.iloc[:,-1].values

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
encode=LabelEncoder()
y[:]=encode.fit_transform(y[:])
y=y.reshape(-1,1)
ohe=OneHotEncoder()
y=ohe.fit_transform(y[:]).toarray()

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(xtrain,ytrain)

ypred=classifier.predict(xtest)