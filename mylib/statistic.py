#
# this is the simple library which will help to find statistic functions.
#

import numpy as np
## This function will give Residual sum of squares (RSS)
##
def rss(coef_, inntercept_, train_x, train_y):
    # print(type(coef_))
    # print(type(inntercept_))
    # print(type(train_x))
    # print(type(train_y))

    rss = np.square(train_y - inntercept_ - coef_ * train_x)
    return np.sum(rss)