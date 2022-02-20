import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    
    TP = len(np.where((y_pred == 1) & (y_true == 1))[0])
    FP = len(np.where((y_pred == 1) & (y_true == 0))[0])
    TN = len(np.where((y_pred == 0) & (y_true == 0))[0])
    FN = len(np.where((y_pred == 0) & (y_true == 1))[0])
    
    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    f1 = 2 * ((precision * recall) / (precision + recall))
    
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    
    return precision, recall, f1, accuracy
    
def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    
    sum_of_id = 0
    for i in range(len(y_true)):
         sum_of_id += len(np.where((y_true == i) & (y_pred == i))[0])
                
    accuracy = sum_of_id / len(y_true)
    
    return accuracy

def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    y_mean = np.mean(y_true)
    up = 0
    down = 0
    for i in range(len(y_true)):
        up += (y_true[i] - y_pred[i]) ** 2
        down += (y_true[i] - y_mean) ** 2
    
    r2 = 1 - (up / down)
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    up = 0
    n = len(y_true)
    for i in range(n):
        up += (y_true[i] - y_pred[i]) **2
    mse = (1/n) * up
    return mse
        
def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    up = 0
    n = len(y_true)
    for i in range(n):
        up += abs(y_true[i] - y_pred[i])
    mae = (1/n) * up
    return mae
    