import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import seaborn
def run():
    mul_var()
    pass

def mul_var():
    data = pd.read_csv('D:\PyCharmWorkspace\Projects\kaggle\data\\adv.csv', index_col='index')
    # plt.plot(data)
    # we will check the model
    tv = data[['TV', 'radio', 'newspaper']]
    sales = data[['sales']]
    train_x, test_x, train_y, test_y = train_test_split(tv, sales)

    # train the model
    model = linear_model.LinearRegression()
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    print("Coefficents : ", model.coef_)
    print("intercept : ", model.intercept_)
    print("mean square : ", mean_squared_error(test_y, predicted))
    print("variace : ", r2_score(test_y, predicted))
    model_1 = sm.OLS(train_y, train_x)
    result = model_1.fit()
    print(result.summary())
    corr_df = train_x.corr(method='pearson')
    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask)] = True
    seaborn.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=1.0, mask= mask, linewidths=2.5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()
    print('Okay!')


