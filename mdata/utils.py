import torch

def one_to_one_hot(batch_size, index, class_num):
    one_hot = torch.Tensor(batch_size, class_num)
    one_hot.zero_()
    indexs = torch.Tensor(batch_size, 1)
    indexs.fill_(index)
    one_hot.scatter_(1, indexs.long(), 1)
    return one_hot

def to_one_hot(batch_size, index, class_num):
    one_hot = torch.Tensor(batch_size, class_num)
    index.resize_(index.shape[0],1)
    one_hot.zero_()
    one_hot.scatter_(1, index, 1)
    return one_hot

def get_half_hot(batch_size, total, one_first):
    first = torch.Tensor(batch_size, total//2).fill_(1)
    second = torch.Tensor(batch_size, total//2).fill_(0)
    if one_first:
        return torch.cat((first, second), 1)
    else:
        return torch.cat((second, first), 1)


if __name__ == "__main__":
    class_num = 10
    batch_size = 4
    label = torch.LongTensor(batch_size, 1).random_() % class_num

    one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
    print(label.shape[0])

    print(one_hot.shape)