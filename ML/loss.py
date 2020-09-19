import numpy as np

# Mean Squared Error (loss function)
def mse(prediction, label):
    loss = sse(prediction, label)
    return loss

# Sum Squared Error (loss function)
def sse(prediction, label):
    loss = np.sum((prediction - label) ** 2)
    return loss

# Root Mean Squared Error (loss function)
def rmse(prediction, label):
    loss = np.sqrt(mse(prediction, label))
    return loss

# Mean Absolute Error (loss function)
def mae(prediction, label):
    loss = np.sum(np.abs(prediction - label))/ len(label)
    return loss

# huber (loss function)
def huber(label, prediction, delta = 1.35):
    loss = np.sum(np.where(np.abs(label-prediction) < delta , 0.5*((label-prediction)**2), delta*np.abs(label - prediction) - 0.5*(delta**2)))
    return loss

# log cosh (loss function)
def logcosh(prediction, label):
    loss = np.sum(np.log(np.cosh(prediction - label)))
    return loss