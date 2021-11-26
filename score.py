#!/usr/bin/env python
# coding: utf-8

# In[51]:


import json
import numpy as np
from azureml.core.model import Model
import joblib

def init():
    global model
    model_path= Model.get_model_path(
        model_name="iris_produccion_model.pkl")
    model = joblib.load(model_path)
    
def run(raw_data):
    data= json.loads(raw_data)["data"]
    data=numpy.array(data)
    result = model.predict(data)
    return {"result":result.tolist()}

init()
test_row = '{"data":[[-43.61941652,   2.04453876],[  6.45467613,   1.15205667], [39.58524735,  -1.48435607], [ 58.55935602,  -0.09698187]]}'

prediction = run(test_row)
print("Test result: ", prediction)

