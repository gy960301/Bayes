import torch

def make_weighted_sum(vw):
    assert len(vw) == 2
    v,w = vw
    assert len(v) == len(w)
    r = sum(v[i] * w[i] for i in range(len(w))) / sum(w)
    return r


def entropy(inputs, reduction="none", binary = True):
    """given a propobility inputs in range [0-1], calculate entroy
    
    Arguments:
        inputs {tensor} -- inputs
    
    Returns:
        tensor -- entropy
    """

    def entropy(p):
        return -1 * p * torch.log(p)

    if binary:
        e = entropy(inputs) + entropy(1 - inputs)
    else:
        e = entropy(inputs)

    if reduction == "none":
        return e
    elif reduction == "mean":
        return torch.mean(e)
    elif reduction == 'sum':
        return torch.sum(e)
    else:
        raise Exception("Not have such reduction mode.")