
import pickle
import pandas as pd


d=pd.read_csv("data.csv")

d1=d.drop(['slope','ca','oldpeak','thal','restecg'],1)

d1=d1.replace({'?':0})

d2=d1.dropna()

x=d2.drop(['target'],1)
y=d2.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
sv=RandomForestClassifier(n_estimators=10,random_state=10,n_jobs=-1).fit(x_train,y_train)


pickle.dump(sv, open('model.pkl' , 'wb'))
model=pickle.load(open("model.pkl",'rb'))