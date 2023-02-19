import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

N = 1000
D = 2

R_inner = 5
R_outer = 10

# we'll use r*sin(theta) + r*cos(theta) to generate points scattered around a circle of radius r
# first, generate the radii - normally distributed around mean r
R1 = np.random.randn(N//2) + R_inner  # need '//', i.e. integer division. The outcome of '/' is always a float
R2 = np.random.randn(N//2) + R_outer

# now generate random angles - centred around 0, but set the standard deviation to 2*pi.
# Otherwise, the standard deviation would be 1, so the points would be mostly between -1 and +1 radian
theta_inner = 2*np.pi*np.random.randn(N//2)
theta_outer = 2*np.pi*np.random.randn(N//2)
X_inner = np.concatenate([[R1*np.cos(theta_inner)], [R1*np.sin(theta_inner)]]).T
X_outer = np.concatenate([[R2*np.cos(theta_outer)], [R2*np.sin(theta_outer)]]).T

X = np.concatenate([X_inner, X_outer])
T = np.array([0]*(N//2) + [1]*(N//2))  # the outer points are class '1'

# the normal logistic regression is not good for this, because it produces a straight line, or a curve, etc.
# in this case, we would have to produce a closed circle to separate these two classes
# instead, we create a column representing the radius of each point:
X = np.c_[np.ones(X.shape[0]), np.concatenate([R1, R2]), X]  # add bias and radii

w = np.random.randn(D+2)  # initialise random weights
Y = sp.expit(X.dot(w))  # get predictions

def cross_entropy(t, y):  # (targets, predictions)
    E = np.mean(-t*np.log(y) - (1-t)*np.log(1-y))  # mean, not sum, because we want this to be independent of N
    return E

# Now do gradient descent to improve the randomly initialised w
entropies = []
eta = 0.0001
steps = 10**3
l2 = 0.1  # l2 regularisation

for i in range(steps):
    w = w - eta*(np.matmul(X.T, Y-T) + l2*w)

    Y = sp.expit(np.dot(X, w))
    entropies.append(cross_entropy(T, Y))

print(f'Final weights: {w}')
print(f'Cross-entropy: {entropies[-1]}')
print(f'Classification rate: {np.mean([T == Y.round()])}')

# we can see that the last two weights are very small - the x and y coordinates do not play a role in the classification
# the weights that truly matter are the bias and the radius
# we can therefore derive the radius of the decision boundary: r = -w0/w1 (which is an approximation neglecting w2, w3)
r = -w[0]/w[1]*np.ones((N, 1))
thetas = np.linspace(0, 2*np.pi, N)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(thetas, r, c='g')
ax.scatter(theta_inner, R1)
ax.scatter(theta_outer, R2)
# ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
ax.set_rlabel_position(90)  # set the positioning of the r label (in radians)
# ax.set_rmax(14)  # set max radius
plt.title('Donut problem')
plt.show()

plt.plot(range(steps), entropies)
plt.xlabel('Iteration')
plt.ylabel('Cross-entropy')
plt.title('Cross-entropy in gradient descent')
plt.show()
