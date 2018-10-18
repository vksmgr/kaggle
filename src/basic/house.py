# require imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

# this is my start function
# main will call this first
def run():
    predict_prise()
    pass

# start the problem
# problem is to predict the housing prise based on the data
# this problem is of the simple linear regression
#

def predict_prise():
    ## first we will explore the data
    data = pd.read_csv("C:\\Users\\205218023\PycharmProjects\kaggle\data\house.csv")
    print(data.head())
    print('Okay!')
    pass