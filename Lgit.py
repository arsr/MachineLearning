from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


X = datasets.load_iris()

data = datasets.load_iris()
X = data.data[:100, :2]
w = np.zeros(X.shape[0])
#self.w = np.zeros((X.shape[1]+1,1))


P = 1 / 1 + np.exp(-np.dot(X, w))

print (P)