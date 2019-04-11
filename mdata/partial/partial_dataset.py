import torch.utils.data as data
import torchvision.datasets as ds
import torch

import torchvision.transforms as transforms

from PIL import Image

import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(dir, class_to_idx, extensions):
    """make a dataset
    
    Arguments:
        dir {string} -- dataset dir
        class_to_idx {dic} -- classed and idx that accept
        extensions {list} -- image extensions list
    
    Returns:
        list -- cosntrcted dataset
    """
    images = []
    accepted = class_to_idx.keys()
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target not in accepted:
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


class PartialDatasetFolder(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        class_to_idx(dic): a dict of class and idx of accepted categories.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(
        self,
        root,
        class_to_idx,
        loader,
        extensions,
        transform=None,
        target_transform=None,
    ):
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            print(class_to_idx)
            print(root)
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + root + "\n"
                    "Supported extensions are: " + ",".join(extensions)
                )
            )

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp,
            self.transform.__repr__().replace("\n", "\n" + " " * len(tmp)),
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp,
            self.target_transform.__repr__().replace(
                "\n", "\n" + " " * len(tmp)
            ),
        )
        return fmt_str


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class PartialImageFolder(PartialDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root,
        class_to_idx,
        transform=None,
        target_transform=None,
        loader=default_loader,
    ):
        super(PartialImageFolder, self).__init__(
            root,
            class_to_idx,
            loader,
            IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
        )
        self.imgs = self.samples


def find_classes(dir):
    classes = [
        d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
    ]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


OFFICE_HOME_CLASS = [
    "Alarm_Clock",
    "Backpack",
    "Batteries",
    "Bed",
    "Bike",
    "Bottle",
    "Bucket",
    "Calculator",
    "Calendar",
    "Candles",
    "Chair",
    "Clipboards",
    "Computer",
    "Couch",
    "Curtains",
    "Desk_Lamp",
    "Drill",
    "Eraser",
    "Exit_Sign",
    "Fan",
    "File_Cabinet",
    "Flipflops",
    "Flowers",
    "Folder",
    "Fork",
    "Glasses",
    "Hammer",
    "Helmet",
    "Kettle",
    "Keyboard",
    "Knives",
    "Lamp_Shade",
    "Laptop",
    "Marker",
    "Monitor",
    "Mop",
    "Mouse",
    "Mug",
    "Notebook",
    "Oven",
    "Pan",
    "Paper_Clip",
    "Pen",
    "Pencil",
    "Postit_Notes",
    "Printer",
    "Push_Pin",
    "Radio",
    "Refrigerator",
    "Ruler",
    "Scissors",
    "Screwdriver",
    "Shelf",
    "Sink",
    "Sneakers",
    "Soda",
    "Speaker",
    "Spoon",
    "TV",
    "Table",
    "Telephone",
    "ToothBrush",
    "Toys",
    "Trash_Can",
    "Webcam",
]

OFFICE_CLASS = [
    "back_pack",
    "bike",
    "bike_helmet",
    "bookcase",
    "bottle",
    "calculator",
    "desk_chair",
    "desk_lamp",
    "desktop_computer",
    "file_cabinet",
    "headphones",
    "keyboard",
    "laptop_computer",
    "letter_tray",
    "mobile_phone",
    "monitor",
    "mouse",
    "mug",
    "paper_notebook",
    "pen",
    "phone",
    "printer",
    "projector",
    "punchers",
    "ring_binder",
    "ruler",
    "scissors",
    "speaker",
    "stapler",
    "tape_dispenser",
    "trash_can",
]


def require_openset_dataloader(
    source_class, target_class, train_transforms, valid_transform, params
):

    source = set(source_class)
    target = set(target_class)
    unknow_idx = len(source)

    source_cls_idx = {sorted(source)[i]: i for i in range(len(source))}
    target_cls_dix = {
        k: source_cls_idx.get(k, unknow_idx) for k in sorted(target)
    }

    dataset = params.dataset
    sourceset = params.source
    targetset = params.target

    source = PartialImageFolder(
        root="./_PUBLIC_DATASET_/" + dataset + "/" + sourceset + "/",
        class_to_idx=source_cls_idx,
        transform=train_transforms,
    )

    target = PartialImageFolder(
        root="./_PUBLIC_DATASET_/" + dataset + "/" + targetset,
        class_to_idx=target_cls_dix,
        transform=train_transforms,
    )

    valid = PartialImageFolder(
        root="./_PUBLIC_DATASET_/" + dataset + "/" + targetset,
        class_to_idx=target_cls_dix,
        transform=valid_transform,
    )

    source = torch.utils.data.DataLoader(
        source,
        batch_size=params.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4,
    )

    target = torch.utils.data.DataLoader(
        target,
        batch_size=params.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4,
    )

    valid = torch.utils.data.DataLoader(
        valid,
        batch_size=params.eval_batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4,
    )

    return source, target, valid

