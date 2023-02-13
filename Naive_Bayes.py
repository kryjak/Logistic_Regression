import numpy as np
import scipy.stats as stats


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
