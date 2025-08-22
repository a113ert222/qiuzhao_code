import math

def dcgatK(rel_scores, k):
    dcg = 0.0
    for i in range(k):
        rel = rel_scores[k]
        dcg += (2**rel-1) / (math.log2(i+1))
    return dcg

def ndcgatK(rel_scores, k):
    dcg = dcgatK(rel_scores, k)
    rel_scores = sorted(rel_scores, reverse=True)
    idcg = dcgatK(rel_scores, k)
    if idcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg