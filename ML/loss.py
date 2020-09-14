import numpy as np

# Mean Squared Error (loss function)
def MSE(prediction, label):
    loss = np.sum((prediction - label) ** 2)/ len(label)
    return loss

# Mean Absolute Error (loss function)
def MAE(prediction, label):
    loss = np.sum(np.abs(prediction - label))/ len(label)
    return loss

# Root Mean Squared Error (loss function)
def RMSE(prediction, label):
    loss = np.sqrt(MSE(prediction, label))
    return loss

# huber loss
def huber(label, prediction, delta = 1.35):
    loss = np.sum(np.where(np.abs(label-prediction) < delta , 0.5*((label-prediction)**2), delta*np.abs(label - prediction) - 0.5*(delta**2)))
    return loss

# log cosh loss
def logcosh(prediction, label):
    loss = np.sum(np.log(np.cosh(prediction - label)))
    return loss