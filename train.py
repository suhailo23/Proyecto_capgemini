#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.utils import shuffle
from sklearn.decomposition import PCA 
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
import joblib

df=pd.read_csv("./iris2.csv")

def split_data(df):
    X=df.columns.values[:-1].tolist()
    Y=df.columns.values[-1:].tolist()
    X,y=shuffle(df[X],df[Y],random_state=10)

    pca_2=PCA(n_components=2)
    X_pca_2=pca_2.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X_pca_2,y,test_size=0.2)
    y_train=y_train.to_numpy().reshape(len(X_train)).astype(int)
    y_test=y_test.to_numpy().reshape(len(y_test)).astype(int)
    data={"train": {"X":X_train, "y":y_train},
         "test":{"X":X_test, "y":y_test}}
    return data
    
def train_model(data, args):
    logistic=LogisticRegression(multi_class="ovr")
    
    grid=GridSearchCV(estimator=logistic, cv=5,param_grid= args)

    grid.fit(data["train"]["X"], data["train"]["y"])
    reg_model=grid.best_estimator_
    return reg_model

def get_model_metrics(reg_model,data):
    y_test_pred = reg_model.predict(data["test"]["X"])
    mean_accuracy =r2_score(data["test"]["y"],y_test_pred)
    metrics={"mean_accuracy":mean_accuracy}
    return metrics

def main():
    sample_data= pd.read_csv("./iris2.csv")
    
    
    data=split_data(sample_data)
    
    
    args={
        "C":[10.0,0.1,0.01,0.001,1e-05],
        "solver":["liblinear","lbfgs"]
         }
    
    reg=train_model(data, args)
    
    
    metrics = get_model_metrics(reg,data)
    
   
    model_name="iris_produccion_model.pkl"
    joblib.dump(value=reg,filename=model_name)
    
main()

