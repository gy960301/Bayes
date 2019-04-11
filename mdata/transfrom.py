import numpy as np
import torchvision.transforms as transforms
import torch


class _ToTensorWithoutScaling(object):
    def __call__(self, picture):
        return torch.FloatTensor(np.array(picture)).permute(2, 0, 1).contiguous()


def get_transfrom(back_bone, is_train):
    if back_bone.upper() in ["ALEX", "ALEXNET"]:

        trans = [
            _ToTensorWithoutScaling(),
            transforms.Normalize(
                mean=[104.0069879317889, 116.66876761696767, 122.6789143406786],
                std=[1, 1, 1],
            ),
        ]

        trans = (
            [
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(227),
            ]
            + trans
            if is_train
            else [transforms.Resize(256), transforms.CenterCrop(227)] + trans
        )

        return transforms.Compose(trans)

    assert False

