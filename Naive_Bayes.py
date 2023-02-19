import numpy as np
import scipy.stats as stats
import scipy.special as sp
import matplotlib.pyplot as plt


"""
Based on the lecture 'Closed-form solution to the Bayes classifier' as well as:
https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Examples

Here, we assume that the likelihoods P(X|Y) for each class are normally distributed, with mean mu_i and covariance 
matrices sigma_i.
If the covariance matrices for each class are identical, this method is known as Linear Discriminant Analysis.
If not, it's called Quadratic Discriminant Analysis.
Additionally, if the covariance matrices are diagonal, we're dealing with a Naive Bayes classifier, which assumes
that the input features are independent.

https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Examples
https://en.wikipedia.org/wiki/Linear_discriminant_analysis
https://en.wikipedia.org/wiki/Quadratic_classifier
"""

# (class (male/female) | height (feet) | weight (lbs) | foot size (inches))
Xtrain = np.array([
    [1, 6.00, 180, 12],
    [1, 5.92, 190, 11],
    [1, 5.58, 170, 12],
    [1, 5.92, 165, 10],
    [0, 5.00, 100, 6],
    [0, 5.50, 150, 8],
    [0, 5.42, 130, 7],
    [0, 5.75, 150, 9]
])

males = Xtrain[Xtrain[:, 0] == 1][:, 1:]
females = Xtrain[Xtrain[:, 0] == 0][:, 1:]

# calculate means and covariances of each class
meanM = np.mean(males, axis=0)
meanF = np.mean(females, axis=0)
# for us, each row is a sample, but np.cov assumes they are in columns
covM = np.diag(np.cov(males.T))  # Naive Bayes assumes that each feature is independent, so the covariance matrix
covF = np.diag(np.cov(females.T))  # should be diagonal. We ignore the off-diagonal terms.

Xtest = np.array([6, 130, 8])

def likelihood(x, mean, cov):
    """ P(X|Y) = GaussianPDF(x, muY, sigmaY)
        data should be a column vector of (height, weight, foot size)^T """
    return stats.multivariate_normal.pdf(x, mean=mean, cov=cov)

def prior(y):
    """ P(Y) with Y = {0, 1} """
    if y == 1:
        return len(males) / len(Xtrain)
    elif y == 0:
        return len(females) / len(Xtrain)
    else:
        print('Wrong category. Should be 1 (male) or 0 (female).')

def posterior(y, x):
    """ P(Y|X) with Y = {0, 1}. Parameters for the likelihood are taken from the training set. """
    # https://stackoverflow.com/questions/43974798/local-variable-might-be-referenced-before-assignment
    ratio = None
    if y == 1:
        ratio = prior(0)*likelihood(x, meanF, covF) / (prior(1)*likelihood(x, meanM, covM))
    elif y == 0:
        ratio = prior(1) * likelihood(x, meanM, covM) / (prior(0) * likelihood(x, meanF, covF))
    else:
        print('Wrong category. Should be 1 (male) or 0 (female).')
    return 1 / (1 + ratio)

print(f'P(1, Xtest) = {posterior(1, Xtest)}')
print(f'P(0, Xtest) = {posterior(0, Xtest)}')  # so the test data point is almost certainly a female

"""
In the below, I tried to treat this data as having come from two 3D Gaussian distributions,
much like what is done at the end of Cross-entropy.py. Here, however, we cannot assume that the two covariances are 
identical, so we have to use the full Quadratic Discriminant Analysis.

Note that in reality, the features that we use here are obviously not independent - we have height, weight and foot 
size. But for Naive Bayes, we have to assume that they're independent and it looks like this leads to some reasonable 
predictions for the test cases.
"""

def QDA(X_train0, X_train1, X_test):
    """ Quadratic Discriminant Analysis """
    # learning the model...
    # means and covariances of all features for both classes
    mean0 = np.mean(X_train0, axis=0)
    mean1 = np.mean(X_train1, axis=0)

    cov0 = np.diag(np.diag(np.cov(X_train0.T)))  # working only with the diagonal entries is compatible with Bayes
    cov1 = np.diag(np.diag(np.cov(X_train1.T)))  # i.e. independent features - it also improves the cross-entropy

    invcov0 = np.linalg.inv(cov0)
    invcov1 = np.linalg.inv(cov1)

    # now get some predictions...
    no0 = len(X_train0)
    no1 = len(X_train1)  # needed to calculate ln(alpha/(1-alpha)), where alpha = P(Y=0) (prior, not likelihood)

    # quadratic, linear and bias terms that go into 1/(1+e^(-y))
    # the three terms are of the form in https://en.wikipedia.org/wiki/Quadratic_classifier
    # diagonal in quadratic is needed because if we have many points in X_test, this term would be NxN, instead of 1xN
    AA = -1/2*invcov0 + 1/2*invcov1
    BB = mean0.dot(invcov0) - mean1.dot(invcov1)
    quadratic = np.diag(X_test.dot(AA).dot(X_test.T))
    linear = BB.dot(X_test.T)
    bias = -1/2*mean0.T.dot(invcov0).dot(mean0) + 1/2*mean1.T.dot(invcov1).dot(mean1) + np.log(no0/no1)
    print(f'quadratic: {-1/2*invcov0 + 1/2*invcov1}')

    y = -(quadratic + linear + bias)  # minus because the full thing is -y
    predictions = sp.expit(y)  # sigmoid

    # now calculate the equation of curve/surface/...

    return predictions

def cross_entropy(t, y):
    E = 0
    for i in range(len(t)):
        if t[i] == 1:
            E += -np.log(y[i])
        elif t[i] == 0:
            E += -np.log(1-y[i])
        else:
            raise Exception('Incorrect target value')
    return E


qda = QDA(females, males, Xtrain[:, 1:])

classes = Xtrain[:, 0]
entropy = cross_entropy(classes, qda)
print(f'Cross-entropy: {entropy}')

plt.scatter(range(len(qda)), classes, c=classes)
plt.plot(range(len(qda)), qda)
plt.text(0, 0.1, f'Cross-entropy: {round(entropy, 3)}', bbox={'facecolor': 'blue', 'alpha': 0.2})
plt.show()

print(QDA(females, males, Xtest.reshape(1, 3)))
