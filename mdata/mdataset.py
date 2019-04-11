import torch.utils.data as data
from PIL import Image

import os

ROOT = "./_PUBLIC_DATASET_/"

def flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)) )				
    return imlist

def load_img(path):
    return Image.open(path).convert('RGB')

class ImageFileListDS(data.Dataset):
    
    def __init__(self, root, flist, transform=None, target_transform=None):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.load_img(os.path.join(self.root, impath))

        if self.transform is not None:
	        img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


if __name__ == "__main__":
    root = r'G:\VS Code\DANN\_PUBLIC_DATASET_\mnist_m\mnist_m_train'
    flist = r'G:\VS Code\DANN\_PUBLIC_DATASET_\mnist_m\mnist_m_train_labels.txt'
    ImageFileList(root, flist)
