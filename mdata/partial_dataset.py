import torch.utils.data as data

class PartialDataset(data.Dataset):

    def __init__(self, dataset, accept_label):
        self.dataset = dataset
        self.accept_label = accept_label

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
