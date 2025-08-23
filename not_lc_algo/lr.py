"""
# 计算预测误差
dz = y_hat - y

# 权重w的梯度
dw = x * dz

# 偏置b的梯度  
db = dz
"""
from torch import sigmoid
import torch.nn as nn
import numpy as np

class logisticRegression:
    def __init__(self, lr=0.01, num_iteration=1000):
        self.lr = lr
        self.num_iter = num_iteration
        self.weight = None
        self.bais = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weight = np.zeros(num_features)
        self.bais = 0

        for _ in range(self.num_iter):
            linear_model = np.dot(X, self.weight) + self.bais
            y_pred = sigmoid(linear_model)

            dw = (1 / num_samples) * (np.dot(X.T, (y_pred - y)))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weight -= self.lr * dw
            self.bais -= self.lr * db
    
    def predict_prob(self, X):
        linear_model = np.dot(X, self.weight) + self.bais
        y_pred = sigmoid(linear_model)
        return y_pred
    
    def predict(self, X, threshold=0.5):
        y_pred_prob = self.predict_prob(X)
        y_pred = np.zeros(y_pred_prob)
        y_pred[y_pred_prob >= threshold] = 1
        return y_pred