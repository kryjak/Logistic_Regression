import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

N = 4
D = 2

# XOR logic gate:
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # input
T = np.array([0, 1, 1, 0])  # output

# how do we separate these two classes?
# again, add one more dimension to the problem
XY = X[:, 0]*X[:, 1]

X = np.c_[np.ones(N), XY, X]  # add bias and X*Y terms

w = np.random.randn(D+2)  # initialise random weights
Y = sp.expit(X.dot(w))  # get predictions

def cross_entropy(t, y):  # (targets, predictions)
    E = np.mean(-t*np.log(y) - (1-t)*np.log(1-y))  # mean, not sum, because we want this to be independent of N
    return E

# Now do gradient descent to improve the randomly initialised w
entropies = []
eta = 0.001
steps = 10**5
l2 = 0.01  # l2 regularisation

for i in range(steps):
    w = w - eta*(np.matmul(X.T, Y-T) + l2*w)

    Y = sp.expit(np.dot(X, w))
    entropies.append(cross_entropy(T, Y))

print(f'Final weights: {w}')
print(f'Cross-entropy: {entropies[-1]}')
print(f'Classification rate: {np.mean([T == Y.round()])}')

# in this case, we have w0 + w1*x*y + w2*x + w3*y = 0 --> y = (-w2*x - w0)/(w1*x + w3)

sing = -w[3]/w[1]  # avoid the singularity
delta = 10**(-3)

xs1 = np.linspace(0, sing-delta)
xs2 = np.linspace(sing+delta, 1)

ys1 = (-w[2]*xs1 - w[0]) / (w[1]*xs1 + w[3])
ys2 = (-w[2]*xs2 - w[0]) / (w[1]*xs2 + w[3])

plt.scatter(X[:, -2], X[:, -1], c=T)
plt.plot(xs1, ys1, color='blue')
plt.plot(xs2, ys2, color='blue')
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.title('XOR logic gate')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
