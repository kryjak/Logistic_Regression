import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

# split the data into two classes
# this creates two 2D Gaussian distributions centred around (-2, -2) and (2, 2)
X[:50] = X[:50] - 2*np.ones((50, D))
X[50:] = X[50:] + 2*np.ones((50, D))

# create some mock target values
T = np.array(50*[0] + 50*[1])

X = np.c_[np.ones(X.shape[0]), X]
# X = np.concatenate((np.ones((N, 1)), X), axis=1)  # less fancy alternative

# use same random weights just to illustrate the concept
w = np.random.randn(D+1)

Y = sp.expit(np.dot(X, w))  # predictions

def cross_entropy(t, y):
    E = 0
    for i in range(Y.shape[0]):
        if t[i] == 1:
            E += -np.log(y[i])
        elif t[i] == 0:
            E += -np.log(1-y[i])
        else:
            raise Exception('Incorrect target value')
    return E

entropy = cross_entropy(T, Y)
print(f'Cross-entropy: {entropy}')

"""
Clearly, the error obtained using random weights is very bad
Let's try the closed-form Bayesian naive classifier.
We use the expression for w^T*x + b from page 5 of the notes, which was obtained from P(Y=1|X):
The mean of the 0-class is (-2, -2), while that of the 1-class is (2, 2).
Because we assume that the features are independent, the covariance matrix should be diagonal.
Moreover, because the samples have been drawn from standard normal, the diagonal values should be 1.
I.e. the covariance matrix should be an identity. Hence, also sigma^(-1) is an identity.
Putting everything together, (mu0 - mu1)*sigma^(-1) = (-4, -4)
The bias term is 0, because the sigma^(-1) terms cancel out (sigma0 = sigma1 and mu0^2 = mu1^2) and
ln(alpha/(1-alpha)) = 0 since alpha = 1/2
Remembering that in the derivation we equated the expression to -(w^T*x + b), we insert an overall minus sign.

The reason why we use the values based on the derivation which had P(Y=1|X) as the starting point - and not also 
values obtained using an analogous derivation which would have P(Y=0|X) as the starting point - is just that the 
whole point of the logistic function is to provide the probability p(x_i) that a given data point x_i belongs to 
class '1'. The probability of it belonging to class '0' is automatically 1-p(x_i) and it is encoded within the 
cross-entropy function. So we do NOT have to use the P(Y=0|X)-based values for the first 50 points, we use the 
P(Y=1|X)-based values for everything.
"""

w = np.array([0, 4, 4])
Y = sp.expit(np.dot(X, w))  # predictions

entropy = cross_entropy(T, Y)
print(f'Cross-entropy: {entropy}')

plt.suptitle('Binary classification on 2D Gaussian data')

plt.subplot(2, 2, 1)
plt.scatter(X[:, 1], X[:, 2], c=T, s=100, alpha=0.5)
# w^T*x + b = [0, 4, 4]*[1, x1, x2] = 4*x1 + 4*x2 --> x2 = -x1
x1 = np.linspace(-6, 6, 100)
x2 = -x1
plt.plot(x1, x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Naive Bayes classifier')

plt.subplot(2, 2, 3)
plt.scatter(range(N), T, c='g', s=20, marker='.')
plt.plot(range(N), Y)
plt.xlabel('Sample point')
plt.ylabel('Logistic function')
plt.title('Targets vs predictions')

"""
Now let's forget about the analytic solution from Naive Bayes and use gradient descent instead
"""
print("GRADIENT DESCENT".center(50, "-"))

eta = 0.1
steps = 25
l2 = 0.1  # l2 regularisation
w = np.random.randn(D+1)  # initialise w - go back to random weights

entropies = []
for i in range(steps):
    Y = sp.expit(np.dot(X, w))
    entropies.append(cross_entropy(T, Y))

    # w = w - eta*np.matmul(X.T, Y-T)
    w = w - eta*(np.matmul(X.T, Y-T) + l2*w)  # now the weights are much smaller

print(f'Cross-entropy: {entropies[-1]}')
print(f'w: {w}')  # resembles [0, 4, 4] from Naive Bayes

plt.subplot(2, 2, 2)
plt.plot(range(steps), entropies)
plt.xlabel('Iteration')
plt.ylabel('Cross-entropy')
plt.title('Cross-entropy in gradient descent')

plt.subplot(2, 2, 4)
plt.scatter(range(N), T, c='g', s=20, marker='.')
plt.plot(range(N), Y)
plt.xlabel('Sample point')
plt.ylabel('Logistic function')
plt.title('Targets vs predictions')

plt.show()
