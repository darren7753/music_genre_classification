import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Note:
# This file is a modified version of the data_loader.py provided 
# as part of the starter code for W2018 cs230 

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.

# In this case, we assume all images have the required sizes. Earlier modules should take care of this!
train_transformer = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor

dev_transformer = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor


class SpecsDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the pngs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.fnames = os.listdir(data_dir)
        self.fnames = [os.path.join(data_dir, f) for f in self.fnames if f.endswith('.png')]

        self.labels = [int(fname.split('/')[-1].split("_")[0]) for fname in self.fnames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.fnames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        im = Image.open(self.fnames[idx])  # PIL image
        im = self.transform(im)
        return im, self.labels[idx]

class TmpSpecsDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    Note that this is a special version of the SpecsDataSet class specified above in the sense that in this particular 
    case, the specs have not yet been classified. As a result this class does not have the labels function

    Another option would be to include conditional statements but it is just cleaner to create a separate class.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the pngs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.fnames = os.listdir(data_dir)
        self.fnames = [os.path.join(data_dir, f) for f in self.fnames if f.endswith('.png')]

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.fnames)

    def __getitem__(self, idx):
        """
        Fetch index idx image from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
        """
        im = Image.open(self.fnames[idx])  # PIL image
        im = self.transform(im)
        return im

def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'dev', 'test', 'tmp' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        dataloaders: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'dev', 'test', 'tmp', 'all']:  
        if split in types:
            path = os.path.join(data_dir, "{}_specs".format(split))

            # use the train_transformer if training data, else use dev_transformer
            if split == 'train':
                dl = DataLoader(SpecsDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            # Use the TmpSpecsDataset class instead of the SpecsDataSet class.
            elif split == 'tmp':
                dl = DataLoader(TmpSpecsDataset(path, dev_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=0,  # Set num_workers to 0 to avoid multiprocessing issues
                                        pin_memory=params.cuda)

            else:
                dl = DataLoader(SpecsDataset(path, dev_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders