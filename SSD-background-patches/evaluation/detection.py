
def accuracy(tp, tn, fp, fn):
    return (tp+tn)/(tp+tn+fp+fn)


def precision(tp, fp):
    return tp/(tp+fp)


def recall(tp, fn):
    return tp/(tp+fn)
