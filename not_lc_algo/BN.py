import numpy as np

def bn(x, gamma, beta, eps=1e-5):
    axis = tuple([0] + list(range(2, x.ndim)))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    x_normalized = (x - mean) / np.sqrt(var)
    reshape_shape = (1, gamma.size) + (1,)*(x.ndim-2) # 让shape为(C,)的gamma变成(1,1,C,1)可以广播的形状，beta同理
    gamma = gamma.reshape(reshape_shape)
    beta = beta.reshape(reshape_shape)
    output = gamma * x_normalized + beta
    return output

x = np.random.randn(2, 3)
gamma = np.ones(3)  # 初始缩放参数为1
beta = np.zeros(3)   # 初始平移参数为0

# 前向传播
output = bn(x, gamma, beta)
print(output.shape)