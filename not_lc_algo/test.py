import itertools
import numpy as np
import torch.nn.functional as F
import torch
from collections import Counter
from torch import sigmoid
import torch.nn as nn
import math

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

# b, c, h, w 对channel这个维度去进行norm
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
    similarity_matrix = torch.matmul(emb1_norm, emb2_norm.T)
    pos_sim = torch.diag(similarity_matrix)
    pos_item = torch.exp(pos_sim / tau)
    neg_item = torch.sum(torch.exp(similarity_matrix / tau), dim=1)
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
        k_label = y_train[k_indice]
        most_common = Counter(k_label).most_common()[0][0]
        predictions.append(most_common)
    return predictions

def kmeans(x, k, thresh=1, max_iter=100):
    centers = x[np.random.choice(x.shape[0], k, replace=False)]
    for i in max_iter:
        distance = np.linalg.norm(x[:, None] - centers, axis=2)
        labels = np.argmin(distance, axis=1)
        new_center = np.array(x[labels == k_i].mean(axis=0) for k_i in range(k))
        if np.linalg.norm(new_center-centers) < thresh:
            break
        centers = new_center
    return labels, centers

class logisticRegression:
    def __init__(self, lr=0.01, num_iterations=1000):
        self.lr = lr
        self.num_iter = num_iterations
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0

        for _ in range(self.num_iter):
            linear_model = np.dot(X, self.w) + self.b
            output = sigmoid(linear_model)
            dw = (1 / num_samples) * (np.dot(X.T, (output-y)))
            db = (1 / num_samples) * np.sum((output-y))
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def predict_prob(self, X):
        linear_model = np.dot(X, self.w) + self.b
        output = sigmoid(linear_model)
        return output
    
    # 二分类问题
    def predict(self, X, thresh=0.5):
        y_probs = self.predict_prob(X)
        y_pred = np.zeros(y_probs)
        y_pred[y_probs > thresh] = 1
        return y_pred
    
class MHA(nn.Module):
    def __init__(self, dim, n_head, dropout=0.1):
        super(MHA, self).__init__()
        self.n_head = n_head
        self.dim = dim
        self.head_dim = self.dim // self.n_head
        assert self.dim == self.head_dim * self.n_head
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.dropout = dropout
        self.fc_out = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        b, t, d = x.size()
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)
        Q = Q.view(b, t, self.n_head, self.head_dim).transpose(1, 2) # b,t,n_h,h_d -> b, n_h, t, h_d
        K = K.view(b, t, self.n_head, self.head_dim).transpose(1, 2) # b,t,n_h,h_d -> b, n_h, t, h_d
        V = V.view(b, t, self.n_head, self.head_dim).transpose(1, 2) # b,t,n_h,h_d -> b, n_h, t, h_d
        score = torch.matmul(Q, K.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_dim), dtype=torch.float32)
        if mask is not None:
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask==0, -1e9) # mask 为0的地方，对应的score变成-1e9
        score = F.softmax(score, dim=-1)
        if self.dropout is not None:
            score = self.dropout(score)
        output = torch.matmul(score, V).transpose(1,2).contiguous().view(b, t, d)
        output = self.fc_out(output)
        return output

def dcgatK(rel_score, k):
    dcg = 0.0
    for i in range(k):
        dcg += (2**rel_score[k] + 1) / (math.log2(i+1))
    return dcg
def ndcgatK(rel_score, k):
    dcg = dcgatK(rel_score, k)
    ideal_rel_score = sorted(rel_score, reverse=True)
    idcg = dcgatK(ideal_rel_score, k)
    if idcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg

# 1->j-1: 大于privot
# j->i-1：小于等于privot
# 初始化：j = l：指针 j指向当前"大于基准值区域"的起始位置
# 遍历完成后：j指向第一个"小于等于基准值"的元素位置
def quick_sort(lst, l, r):
    if l >= r:
        return
    idx = patition(lst, l, r)
    quick_sort(lst, l, idx-1)
    quick_sort(lst, idx+1, r)
def patition(lst, l, r):
    privot = lst[r]
    j = l
    for i in range(l,r):
        if lst[i] > privot:
            lst[i], lst[j] = lst[j], lst[i]
            j += 1
    lst[j], lst[r] = lst[r], lst[j]
    return j

# logits: n_samples, n_classes
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return softmax_probs