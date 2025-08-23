import itertools
import numpy as np
import torch.nn.functional as F
import torch
from collections import Counter

def auc(preds, labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    res = 0
    acc_neg = 0
    lst = sorted(zip(labels, preds), key=lambda x:x[1])
    for pred, pairs in itertools.groupby(lst, key=lambda x: x[1]):
        pair_cnt = 0
        pos_cnt = 0
        for label, _ in pairs:
            pair_cnt += 1
            if label == 1:
                pos_cnt += 1
        res += pos_cnt * acc_neg + pos_cnt * (pair_cnt - pos_cnt) * 0.5
        acc_neg += (pair_cnt - pos_cnt)
    return res * 1.0 / (n_neg * n_pos)

# b, c, h, w
def BN(x, gamma, beta):
    axis = tuple([0] + list(range(2, x.ndim)))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var)
    reshape_shape = (1, gamma.size) + (1,) * (x.ndim-2)
    gamma = gamma.reshape(reshape_shape)
    beta = beta.reshape(reshape_shape)
    output = gamma * x_norm + beta
    return output

def ce(y_pred, y_label, epsilon=1e-12):
    y_pred= np.clip(y_pred, epsilon, 1-epsilon)
    loss = - np.sum(y_label * np.log(y_pred), axis=1)
    return np.mean(loss)
    
def calCL(Emb1, Emb2, idx, tau):
    emb1 = F.embedding(idx, Emb1)
    emb2 = F.embedding(idx, Emb2)
    emb1_norm = F.normalize(emb1, p=2, dim=-1)
    emb2_norm = F.normalize(emb2, p=2, dim=-1)
    similarity_matrix = torch.matmul(emb1_norm, emb2_norm)
    pos_sim = torch.diag(similarity_matrix)
    pos_item = torch.exp(pos_sim / tau)
    neg_item = torch.sum(torch.exp(similarity_matrix / tau))
    loss = -torch.sum(torch.log(pos_item / neg_item))
    return loss

## 堆排序
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[largest], arr[i] = arr[i], arr[largest]
        heapify(arr, n, largest)


def heap_sort(arr):
    n = len(arr)
    for i in range(n//2-1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

def kaigenhao(x):
    if x < 0:
        return False
    if x == 1:
        return 1
    guess = x / 2
    while True:
        new_guess = (guess + x / guess) / 2
        if abs(new_guess - guess) < 1e-9:
            return new_guess
        guess = new_guess

def knn(x_train, y_train, x_test, k=3):
    predictions = []
    for x in x_test:
        distance = np.linalg.norm(x_train-x, axis=1)
        k_indice = np.argsort(distance)[:k]
        k_label = x_train[k_indice]
        most_common = Counter(k_label).most_common()[0][0]
        predictions.append(most_common)
    return predictions

def kmeans()