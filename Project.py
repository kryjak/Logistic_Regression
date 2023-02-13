"""
In this project, we will simulate an e-commerce data analysis.
Using data on customers visiting a website, we will try to predict the outcome of the visit.

The CSV data provided has the following columns:
- Is_mobile (0/1) <--- we will use just one column with 1 or 0, not [1 0] or [0 1]
- N_products_viewed (int >= 0)
- Visit_duration (real >= 0)
- Is_returning_visitor (0/1) - same as above
- Time_of_day (0/1/2/3) - day divided into 6hr slots <--- we will use 'one-hot encoding)
- User_action (bounce/add_to_cart/begin_checkout/finish_checkout)

In the logistic regression class, we will only do a binary classification, i.e. we will just try to predict
'bounce' or 'add_to_cart'
In NN class, we can have multi-class classification, but we will not cover it here.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt



"""
First, we do pre-processing on the CSV data
"""

df = pd.read_csv('ecommerce_data.csv')

print(df.columns)
# df['time_of_day'].hist()
# plt.show()

data = df.to_numpy()
np.random.shuffle(data)  # why???

X, Y = data[:, :-1], data[:, -1].astype(np.int32)
X = np.c_[np.ones(data.shape[0]), X]  # should we be adding a column of 1s here?
N, D = X.shape
print(f'(N, D) = {N, D}')

# one-hot encoding of time_of_day:
onehot = np.zeros((N, 4))  # four entries because we have four categories: 0/1/2/3, so we need [1, 0, 0, 0], ...
onehot[np.arange(N), X[:, -1].astype(np.int32)] = 1
# note the difference between onehot[np.arange(N), [...]] and onehot[:, [...]]

# alternatively, we can just use a for-loop, but it's slower:
# onehot = np.zeros((N, 4))
# for ii in range(N):
#     onehot[ii, X[ii, -1].astype(np.int32)] = 1

# now, we concatenate onto the non-categorical columns:
X = np.concatenate((X[:, :-1], onehot), axis=1)

# split into train and test sets:
Xtrain, Ytrain = X[:-100], Y[:-100]
Xtest, Ytest = X[-100:], Y[-100:]

# now we normalise the columns 1 and 2
for col in (1, 2):
    m = Xtrain[:, col].mean()
    s = Xtrain[:, col].std()

    Xtrain[:, col] = (Xtrain[:, col]-m)/s
    Xtest[:, col] = (Xtest[:, col]-m)/s  # why are we dividing by m and s of the Xtrain here?

# for binary classification, we will just get the data for which the output is either 0 or 1 (the first two outcomes)
Xbintrain = Xtrain[Ytrain <= 1]  # slicing based on Boolean input - nice!
Ybintrain = Ytrain[Ytrain <= 1]
Xbintest = Xtest[Ytest <= 1]
Ybintest = Ytest[Ytest <= 1]

print(Xbintrain.shape, Ybintrain.shape, Xbintest.shape, Ybintest.shape)

"""
Now we will use logistic regression to make prediction.
We haven't trained the model yet, so the predictions are going to use random weights w.
"""
# randomly initialise weights
w = np.random.randn(Xbintrain.shape[1])

def forward(X, w):
    return expit(X.dot(w))

PofYgivenX = forward(Xbintrain, w)
print(PofYgivenX.shape)

predictions = PofYgivenX.round().astype(np.int32)
# predictions = [1 if ii >= 0.5 else 0 if ii <= 0.5 else ii for ii in PofYgivenX]  # same thing

classification_rate = np.mean(Ybintrain == predictions)  # mean(array of True or False) is like mean(0s or 1s)
print('Score:', classification_rate)
