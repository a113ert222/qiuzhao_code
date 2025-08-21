import itertools

def auc(labels, preds):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    lst = sorted(zip(labels, preds), lambda x:x[1])
    acc_neg = 0
    res = 0
    for preds, pairs in itertools.groupby(lst, key=lambda x:x[1]):
        pair_cnts = 0
        pos_cnt = 0
        for label, pred in pairs:
            pair_cnts += 1
            if label == 1:
                pos_cnt += 1
        res += pos_cnt * acc_neg + pos_cnt * (pair_cnts - pos_cnt) * 0.5
        acc_neg += (pair_cnts - pos_cnt)
    return res * 1.0 / (n_neg * n_pos)