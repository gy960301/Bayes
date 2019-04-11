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
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
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
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


def indepedent_class_seperation(source_class_idx):

    d_1 = {
        "back_pack": 0,
        "bike": 1,
        "bike_helmet": 2,
        "bookcase": 3,
        "bottle": 4,
        "calculator": 5,
        "desk_chair": 6,
        "desk_lamp": 7,
        "desktop_computer": 8,
        "file_cabinet": 9,
        "headphones": 10,
        "keyboard": 11,
        "laptop_computer": 12,
        "letter_tray": 13,
    }
    c1 = ClassSeretation(d_1, ("W",))

    d_2 = {
        "mobile_phone": 14,
        "monitor": 15,
        "mouse": 16,
        "mug": 17,
        "paper_notebook": 18,
        "pen": 19,
        "phone": 20,
        "printer": 21,
    }
    c2 = ClassSeretation(d_2, ("A", "W"))

    d_3 = {
        "projector": 22,
        "punchers": 23,
        "ring_binder": 24,
        "ruler": 25,
        "scissors": 26,
        "speaker": 27,
        "stapler": 28,
        "tape_dispenser": 29,
        "trash_can": 30,
    }
    c3 = ClassSeretation(d_3, ("A",))

    c = (c1, c2, c3)

    return c


class ClassSeretation(object):
    def __init__(self, classes_to_idx, domains):
        self.classes = [i for i, _ in classes_to_idx.items()]
        self.classes_to_idx = classes_to_idx
        self.domains = domains
        self.domain_loader = dict()
    
    def set_domain_loader(self, domain, loader):
        self.domain_loader[domain] = loader
    
    def get_domain_loader(self, domain):
        return self.domain_loader[domain]


class MultiFolderDataHandler(object):
    def __init__(self, root, sources, target, params):
        """A handler that use for multi source domain adaptation with category shift, which generate a indenpdent class
        
        an indenpdent class group is a list of struct
        (classes,(domain_name, (participate examples set, dataloader)))

        Arguments:
            root {string} -- data set path
            sources {list} -- source dataset names
            target {string} -- target dataset name

        
        Examples:

            mhandler = MultiFolderDataHandler(
                    root="Office_Shift", sources=["A", "W"], target="D"
            )
            icgs = mhandler.seperation_with_loader()

            for categories, participator in icgs:
                print(categories.values())
                for domain, data in participator.items():
                    dataset, dataloader = data
                    print(domain)
        
        """

        self.params = params

        root = "./_PUBLIC_DATASET_/" + root + "/"
        self.root = root

        trans = [transforms.Resize(224), transforms.ToTensor()]
        self.transforms = transforms.Compose(trans)

        target_sets = ds.ImageFolder(root=root + target, transform=self.transforms)
        total_classes = target_sets.class_to_idx
        target_set = target_sets
        self.total_classes = total_classes
        self.target_set = target_set

        source_classes = dict()
        for i in sources:
            s = self.remap_source_target(i)
            source_classes[i] = s

        ics = indepedent_class_seperation(source_classes)
        self.independ_class_seperation = ics


    def remap_source_target(self, source):
        root = self.root + source

        old_class_idx = find_classes(root)
        old_class_idx = {y: x for x, y in old_class_idx.items()}

        new_class_idx = dict()
        for _, class_name in old_class_idx.items():
            target = self.total_classes[class_name]
            new_class_idx[class_name] = target

        return new_class_idx

    def seperation_with_loader(self, shuffle=True, drop_last=False):
        """ make a independed class seperation
         
        Returns:
            [dict] -- a dict of {class : {domain: dataloader}}
        """
        transforms = self.transforms
        batch_size = self.params.batch_size
        root = self.root

        for seperation in self.independ_class_seperation:
            icg = dict()
            for domain in seperation.domains:
                dataset = PartialImageFolder(
                    root=root + domain,
                    class_to_idx=seperation.classes_to_idx,
                    transform=transforms,
                )
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
                )
                seperation.set_domain_loader(domain, dataloader)

        target_loader = torch.utils.data.DataLoader(
            self.target_set, batch_size=batch_size, shuffle=True, drop_last=drop_last
        )

        valid_loader = torch.utils.data.DataLoader(
            self.target_set, batch_size=batch_size, shuffle=True, drop_last=drop_last
        )

        return self.independ_class_seperation, target_loader, valid_loader


if __name__ == "__main__":

    mhandler = MultiFolderDataHandler(
        root="Office_Shift", sources=["A", "W"], target="D"
    )
    ics = mhandler.independ_class_seperation
    # print(ics)

    # class_idx = [j for j, _ in ics]
    # class_group = [j for j in [list(i.keys()) for i in class_idx ]]
    # class_group_idx = [i for i in range(len(class_group))]

    # import itertools
    # current_idx = itertools.cycle(iter(class_group_idx))

    icgs, target_loader, valid_loader = mhandler.seperation_with_loader()

    # a dict of {categorys: {domain : loader}}
    partial_loaders = list()

    for idx, (_, participator) in enumerate(icgs):
        domain_it = dict()
        for domain, data in participator.items():
            dataset, dataloader = data
            """ init a predict unit
                added to union
            """
            domain_it[domain] = dataloader
        partial_loaders[idx] = domain_it

    iters = dict()
    mode = "train"

    source_iters = list()
    for idx, domain_loaders in enumerate(partial_loaders):
        for domain, loaders in domain_loaders.items():
            source_iters[idx][domain] = loaders

