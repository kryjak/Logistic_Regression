"""Facial recognition project"""

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from util import getData, getBinaryData, sigmoid_cost, error_rate
from sklearn.utils import shuffle  # shuffles the data in multiple arrays in the same order

"""
Obviously, computers do not understand emotions on human faces - but they can recognise patterns in pixels.
Image pixels can be considered as input features to our model.
The code is going to be class-based to mimic the Scikit Learn interface.
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
48x48 greyscale images of faces that have been pre-processed so that each face is centered and occupies roughly the 
same area of an image.
Each face expresses an emotion:
0 - angry
1 - disgust
2 - fear
3 - happy
4 - sad
5 - surprise
6 - neutral
Data is given as CSV: label, pixels (space-separated), train/test

We will only choose class 0 and 1 to make it into a binary classification problem.
Another important distinction is that we will ignore any spatial relations between pixels, i.e. we will flatten the 
matrix into an array of length 48x48=2304. That is, the fact that, e.g. pixel 1 is directly above pixel 49 is 
completely ignored and each pixel is treated independently. This is the case for logistic regression and basic neural 
networks, but with CNNs, we keep the matrix form and do take into account the spatial relationships.

Note that this leads to a certain problem: class 0 has 4953 samples, class 1 has 547 samples.
This sort of a problem appears a lot in e.g. medical testing. For example if only 1% of the population has some rare 
condition, then the corresponding class is going to be vastly underrepresented.
This is a problem since a random classifier would be correct 99% of the time. Alternatively, a classifier could 
predict 'no disease' every time and it would also be 99% correct.
There are two ways to deal with this problem. Say we have 1000 samples from class 0, 100 samples from class 1:
1) Keep only 100 from class 0, now we have 100 vs 100.
2) Repeat class 1 10 times, now we have 1000 vs 1000.
These two strategies has the same expected error rate, but we expect method 2) to perform better as it is trained on 
a larger amount of data. 
We can also expand class 1 by:
- adding a bit of Gaussian noise to samples from class 1, but not too much! Otherwise we'll modify the sample too much 
such that it might not realistically even belong to class 1.
- apply an invariant transformation (shift left/right, etc)

For reasons explained in the course earlier in the course, we want to normalise the data to be between 0 and 1 (
roughly). So we will divide 0...255 by 255. (We could also subtract the mean and divide by the std, but since the 
data is non-negative, we can just divide by the max value.) Another reasons why we want data to be normalised between 
0 and 1 is that this is precisely the region where the logistic function (sigmoid/tanh/etc) is most sensitive.
"""

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# display the data to see what kind of images we're working with
def show_images():
    X, Y, _, _ = getData(balance_ones=False)

    # for each emotion, choose a random image, reshape it into a 48x48 array and display
    while True:
        for i in range(len(label_map)):
            x, y, = X[Y == i], Y[Y == i]
            N = len(y)
            j = np.random.choice(N)
            plt.imshow(x[j].reshape(48, 48), cmap='gray')
            plt.title(label_map[y[j]])
            plt.show()

        # stop displaying when prompted by the user
        prompt = input('Quit? Enter Y:\n')
        if prompt == 'Y':
            break

show_images()

# define the logistic model as a class
class LogisticModel(object):
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, Y, learning_rate=10e-7, l2=0, steps=120000, show_fig=False):
        # shuffle the data so that the train and test sets contain a representative sample of each class
        X, Y = shuffle(X, Y)
        # split into test and train (validation) sets
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        self.w = np.random.randn(D) / np.sqrt(D)  # initialise w to random values from Gaussian with std = sqrt(D)
        self.b = 0

        costs = []
        best_validation_error = 1

        for i in range(steps):
            pY = self.forward(X)  # get the output of sigmoid

            # gradient descent
            self.w = self.w - learning_rate * (np.matmul(X.T, pY - Y) + l2 * self.w)
            self.b = self.b - learning_rate * ((pY - Y).sum() + l2 * self.b)

            if i % 20 == 0:
                pYvalid = self.forward(Xvalid)
                c = sigmoid_cost(Yvalid, pYvalid)  # cross-entropy
                costs.append(c)
                e = error_rate(Yvalid, pYvalid.round())  # 1 - classification_rate
                if e < best_validation_error:
                    best_validation_error = e  # update the best test error

        print(f'best_validation error: {best_validation_error}')

        if show_fig:
            plt.plot(costs)
            plt.show()

    # logistic sigmoid function
    def forward(self, X):
        return sp.expit(X.dot(self.w) + self.b)

    # same, but rounded to 0 and 1, so the actual prediction for the class
    def predict(self, X):
        pY = self.forward(X)
        return pY.round()

    # classification_rate = 1 - error_rate
    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)

def main():
    X, Y = getBinaryData()  # like getData, but extracts only class 0 and 1

    X0 = X[Y == 0, :]  # class 0
    X1 = X[Y == 1, :]  # class 1

    X1 = np.repeat(X1, 9, axis=0)  # repeat class 1 to equalise the number of samples with class 0
    X = np.vstack([X0, X1])
    Y = np.array([0] * len(X0) + [1] * len(X1))  # adjust the array of targets accordingly

    # scikit-like syntax: define the model, train, predict, asses the score
    model = LogisticModel()
    model.fit(X, Y, steps=1000, show_fig=True)
    model.score(X, Y)

if __name__ == '__main__':
    main()
