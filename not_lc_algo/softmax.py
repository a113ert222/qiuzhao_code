import numpy as np

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits,axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return softmax_probs