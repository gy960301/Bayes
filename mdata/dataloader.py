import torchvision.datasets as ds
import torch.utils.data

import torchvision.transforms as transforms

import torchvision.transforms.transforms as transforms

import numpy as np
import torchvision

from enum import Enum

import os

DS = ["SVHN", "MNIST", "OFFIE"]


class DSStastic:
    """Mean and Std for very Datasets
    """

    SVHN = np.array(([0.44, 0.44, 0.44], [0.19, 0.19, 0.19]))

    MNIST = np.array(([0.44, 0.44, 0.44], [0.19, 0.19, 0.191]))


class _ToTensorWithoutScaling(object):
    def __call__(self, picture):
        return torch.FloatTensor(np.array(picture)).permute(2, 0, 1).contiguous()



def get_dataset(dsname, domain=None, split="train", size=224):
    """Helpper function to get `DataLoader` of specific datasets 
    
    Arguments:
        name {DSNames} -- data set
        param {[type]} -- parameters to generate dataloader
            - `param.batch_size` to spectic batch size.
    
    Keyword Arguments:
        root {str} -- [root path for dataset] (default: {'./data'})
        split {str} -- [get 'train' or 'valid' part] (default: {'train'})
        dowloard {bool} -- [if not exits, want to dowload?] (default: {False})
        mode {str} -- ['norm' the dataset or 'rerange' to [-1,1]] (default: {'norm'})
    
    Returns:
        [DataLoader] -- [a DataLoader for the dataset]
    """

    if split not in ["train", "test", "valid"]:
        raise Exception("Not support " + str(split))

    #########################################
    #! Prepare dataset translation
    #########################################

    ## UGLY need optim

    if dsname in [None, "NONE"]:
        dsname = domain

    assert dsname.upper() in ["MNIST", "SVHN", "OFFICE"]

    if dsname in ["MNIST", "MNIST"]:
        resize = 28
        crop = 28

        trans = [
            transforms.Resize(crop),
            transforms.ToTensor(),
        ]
    else:
        resize = 256
        crop = 227
        # 224 for resnet
        # 227 for alexnet

        mean_color = [
            104.0069879317889 ,
            116.66876761696767 ,
            122.6789143406786 ,
        ]

        op_toTensor = _ToTensorWithoutScaling() if True else transforms.ToTensor()

        if split == "train":
            trans = [
                transforms.Resize(crop),
                transforms.RandomHorizontalFlip(),
                _ToTensorWithoutScaling(),
                transforms.Normalize(
                    mean=mean_color, std=[1, 1, 1]
                ),
            ]
        else:
            trans = [
                transforms.Resize(resize),
                transforms.CenterCrop(crop),
                _ToTensorWithoutScaling(),
                transforms.Normalize(
                    mean=mean_color, std=[1, 1, 1]
                ),
            ]
    
    transform = transforms.Compose(trans)

    #########################################
    #! Fetching dataset
    #########################################
    root = "./_PUBLIC_DATASET_/"

    ## MNIST dataset
    if dsname == "MNIST":
        train = split == "train"
        data_set = ds.MNIST(
            root=root, train=train, transform=transform, download=True
        )
    ## SVHN dataset
    elif dsname == "SVHN":
        data_set = ds.SVHN(
            root=root + "SVHN/",
            split=split,
            transform=transform,
            download=True,
        )
    ## OFFICE dataset
    elif dsname.upper() == "OFFICE":
        if domain not in ["A", "D", "W"]:
            raise Exception(str(domain) + " not in OFFICE dataset.")
        else:
            data_set = ds.ImageFolder(
                root=root + "Office/" + domain, transform=transform
            )

    ## OFFICE dataset
    elif dsname == "OfficeHome":
        if domain not in ["Ar", "D", "W"]:
            raise Exception(str(domain) + " not in OFFICE dataset.")
        else:
            data_set = ds.ImageFolder(
                root=root + "OfficeHome/" + domain, transform=transform
            )

    else:
        raise Exception(str(dsname) + " Not Support")

    return data_set  # , data_loader


def load_img_dataset(
    dataset, subset, batch_size, test=False, target_transform=None
):

    if test:
        trans_corp = [transforms.Resize(256), transforms.CenterCrop(224)]
    else:
        trans_corp = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]

    trans_tensor = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    transform = transforms.Compose(trans_corp + trans_tensor)

    dataset = ds.ImageFolder(
        root="./_PUBLIC_DATASET_/" + dataset + "/" + subset,
        transform=transform,
        target_transform=target_transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return dataset, data_loader


# class RemappingIndex(object):
#     def __init__(self, old, new):
#         self.old = {y: x for x, y in old.items()}
#         self.new = new

#     def __call__(self, target):
#         class_name = self.old[target]
#         target = self.new[class_name]
#         return target

# def find_classes(dir):
#     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return class_to_idx

# def indepedent_class_group(source_class_idx):

#     d_1 = {'back_pack': 0, 'bike': 1, 'bike_helmet': 2, 'bookcase': 3, 'bottle': 4, 'calculator': 5, 'desk_chair': 6, 'desk_lamp': 7, 'desktop_computer': 8, 'file_cabinet': 9, 'headphones': 10, 'keyboard': 11, 'laptop_computer': 12, 'letter_tray': 13}
#     c_1 = (d_1, ('W'))

#     d_2 = {'mobile_phone': 14, 'monitor': 15, 'mouse': 16, 'mug': 17, 'paper_notebook': 18, 'pen': 19, 'phone': 20, 'printer': 21,}
#     c_2 = (d_2, ('A','W'))

#     d_3 = {'projector': 22, 'punchers': 23, 'ring_binder': 24, 'ruler': 25, 'scissors': 26, 'speaker': 27, 'stapler': 28, 'tape_dispenser': 29, 'trash_can': 30}
#     c_3 = (d_3, ('A'))

#     c = (c_1, c_2, c_3)

#     return c

# class MultiFolderDataHandler(object):
#     def __init__(self, root, sources, target, transform=None):

#         root = "./_PUBLIC_DATASET_/" + root + "/"
#         self.root = root

#         target_sets = ds.ImageFolder(root=root + target, transform=transform)
#         total_classes = target_sets.class_to_idx
#         target_set = target_sets
#         self.total_classes = total_classes
#         self.target_set = target_set

#         source_classes = dict()
#         for i in sources:
#             s = self.remap_source_target(i)
#             source_classes[i] = s

#         icg = indepedent_class_group(source_classes)
#         print(icg)


#     def remap_source_target(self, source):
#         root = self.root + source

#         old_class_idx = find_classes(root)
#         old_class_idx = {y: x for x, y in old_class_idx.items()}

#         new_class_idx = dict()
#         for _, class_name in old_class_idx.items():
#             target = self.total_classes[class_name]
#             new_class_idx[class_name] = target

#         return new_class_idx


if __name__ == "__main__":

    # mhandler = MultiFolderDataHandler(
    #     root="Office_Shift", sources=["A", "W"], target="D"
    # )

    # def find_classes(dir):
    #     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    #     classes.sort()
    #     class_to_idx = {classes[i]: i for i in range(len(classes))}
    #     return classes, class_to_idx

    # def remap_source_class_idx(root, sources, target):
    #     classes, _ = find_classes("./_PUBLIC_DATASET_/Office_Shift/A")
    #     print(classes)

    # print(O_D.class_to_idx)

    dataset = ds.ImageFolder(
        root="./_PUBLIC_DATASET_/" + "Office_Shift" + "/" + "A"
    )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    import numpy as np

    it = np.nditer(data_loader)
    # it = iter(data_loader)
    print(next(it)[1])
