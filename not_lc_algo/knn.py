import numpy as np
from collections import Counter

def knn_predictor(x_train, y_train, x_test, k=3):
    predictions = []
    for x in x_test:
        distance = np.linalg.norm(x_train-x, axis=1)
        k_indices = np.argsort(distance)[:k]
        k_labels = y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0] # .most_common(1)：返回一个列表，里面包含出现次数最多的1个元素及其次数，格式如 [(label, count)]。
        predictions.append(most_common)
    return np.array(predictions)

# 测试示例
if __name__ == "__main__":
    # 训练数据
    X_train = np.array([[1, 1], [2, 2], [3, 3]])
    y_train = np.array([0, 0, 1])
    
    # 测试数据
    X_test = np.array([[1.5, 1.5], [2.5, 2.5]])
    
    # 预测并打印结果
    preds = knn_predictor(X_train, y_train, X_test, k=2)
    print("预测结果:", preds)  # 输出: [0 0]（注意平局情况可能与实际预期不同）