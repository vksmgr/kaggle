# require imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
    data = pd.read_csv("D:\PyCharmWorkspace\Projects\kaggle\data\house.csv")
    # first we will visualize the data
    # print(data.columns)
    # first we will select the feature which we will be using
    sqft_living = np.array(data['sqft_living']).reshape((-1, 1))
    price = np.array(data['price'])

    # plt.scatter(sqft_living, price, color='red')
    # plt.show()
    # now we will split the data into train and test module
    train_x, test_x, train_y, test_y = train_test_split(sqft_living, price, test_size=0.20)

    #now we will create the linear regression model
    model = linear_model.LinearRegression()

    #train your model
    model.fit(train_x, train_y)

    #test your model
    predict_prise = model.predict(test_x)

    #measure the performance
    print("coefficient: {}".format(model.coef_))
    print("intercept: {}".format(model.intercept_))

    # mean square error
    print("mean square : {}".format(mean_squared_error(test_y, predict_prise)))
    print("Variace : {}".format(r2_score(test_y, predict_prise)))


    #test values
    model_1 = sm.OLS(train_y, train_x)
    result = model_1.fit()
    print(result.summary())

    #ploting the result
    # plt.scatter(test_x, test_y, color='red')
    # plt.plot(test_x, predict_prise, linewidth=3)
    # plt.show()
    print('Okay!')
    pass