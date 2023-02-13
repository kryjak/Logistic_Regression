import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

# just like in linear regression, add a column of 1s to account for the bias term
X = np.c_[np.ones(X.shape[0]), X]
# X = np.concatenate((np.ones((N, 1)), X), axis=1)  # less fancy alternative

w = np.random.randn(D+1)

Z = np.dot(X, w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

print(sigmoid(Z))
# we can also use scipy's builtin 'expit' (don't know why the name is so strange, but it is the logistic function)
# print(sp.expit(Z))

plt.plot(sigmoid(Z))
plt.show()
