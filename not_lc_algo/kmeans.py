import numpy as np

def kmeans(x, k, thresh=1, max_iter=100):
    centers = x[np.random.choice(x.shape[0], k, replace=False)]
    for i in range(max_iter):
        # x[:, None] n,f -> n, 1, f
        # center k, f
        #  -> n, k, f
        # 每个点都与簇中心做减法
        distance = np.linalg.norm(x[:, None] - centers, axis=2)
        labels = np.argmin(distance, axis=1)
        new_centers = np.array(x[labels==k_i].mean(axis=0) for k_i in range(k)) # 对每个特征维度分别求平均，得到一个新的中心点坐标。
        if np.linalg.norm(centers - new_centers) < thresh:
            break
        centers = new_centers
    return labels, centers 

X = np.random.rand(100,2)
k = 3
labels, centers = kmeans(X, k)
print("簇标签", labels)
print("聚类中心点", centers)