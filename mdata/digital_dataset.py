from enum import Enum

import torchvision.datasets as torchds
from torch.utils.data.dataset import ConcatDataset

from mdata.mdataset import ROOT
from mdata.mdataset import ImageFileListDS
from mdata.USPS import USPS


class DigitalNames(Enum):
    MNIST = "MNIST"
    SVHN = "SVHM"
    MNIST_M = "MNIST-M"
    USPS = "USPS"


def get_digital_dataset(
    dsnames, train=True, transform=None, target_transform=None
):
  
    if not isinstance(dsnames, (list, tuple)):
        dsnames = (dsnames,)
       
    result = list()
    for ds in dsnames:
        if ds == "MNIST":
            dataset = torchds.MNIST(
                root=ROOT + "MNIST",
                train=train,
                transform=transform,
                target_transform=target_transform,
                download=True,
            )
        elif ds == "SVHN":
            dataset = torchds.SVHN(
                root=ROOT + "SVHM",
                split="train" if train else "test",
                transform=transform,
                target_transform=target_transform,
                download=True,
            )
        elif ds == "MNIST-M":
            if train:
                list_name = r"mnist_m/mnist_m_train_labels.txt"
                image_folder = r"mnist_m/mnist_m_train"
            else:
                list_name = r"mnist_m/mnist_m_test_labels.txt"
                image_folder = r"mnist_m/mnist_m_test"
            dataset = ImageFileListDS(
                root=ROOT + image_folder,
                flist=ROOT + list_name,
                transform=transform,
                target_transform=target_transform,
            )
        elif ds == "USPS":
            dataset = USPS(
                root=ROOT + "USPS", train=train, transform=transform, download=True
            )
        else:
            dataset = None
            print(ds)
            raise Exception("no data set " + ds)

        if dataset is not None:
            result.append(dataset)
    
    return result[0] if len(result) == 1 else result

