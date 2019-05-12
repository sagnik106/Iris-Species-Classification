import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
import numpy as np
import pandas as pd

data=pd.read_csv("Iris.csv")
x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
e=LabelEncoder()
y[:]=e.fit_transform(y[:])
y=y.reshape(-1,1)
ohe=OneHotEncoder()
y=ohe.fit_transform(y[:]).toarray()

def deepl():
    model=Sequential()
    s=x.shape[1]
    s1=y.shape[1]
    model.add(Dense(s,input_dim=s,activation='relu'))
    #model.add(Dense(s,activation='relu'))
    model.add(Dense(s1,activation='softmax'))
    model.compile(Adam(lr=0.1),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

model=deepl()
pm=model.fit(x,y,batch_size=10,epochs=5,verbose=1,validation_split=0.1)
