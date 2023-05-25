import xarray as xr
import numpy as np
import pandas as pd
import torch
import pathlib


def cloudcatalogue_find(root, dataset):
    """
    Retrieve the file paths for CloudCatalogue dataset.

    Args:
        root (pathlib.Path): The root directory of the dataset.
        dataset (pd.DataFrame): The dataset containing scene information.

    Returns:
        tuple: A tuple containing the X and y file paths.

    """
    Xroot = root / "subscenes"
    yroot = root / "masks"    
    Xfiles = []
    yfiles = []
    
    for item in range(len(dataset)):
        if dataset.shadows_marked[item] == 1:
            Xfiles.append(Xroot / (dataset.scene[item] + ".npy"))
            yfiles.append(yroot / (dataset.scene[item] + ".npy"))    
    return Xfiles, yfiles


class CloudDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        """
        CloudCatalogue dataset class for loading and preprocessing data.

        Args:
            root (pathlib.Path): The root directory of the dataset.
            dataset (pd.DataFrame): The dataset containing scene information.

        """
        self.cloudcataloguefiles = files
        self.X = self.cloudcataloguefiles[0]
        self.y = self.cloudcataloguefiles[1]
        
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple: A tuple containing the input and target data.

        """
        Xfile = self.X[idx]
        yfile = self.y[idx]
        
        # Add padding to the input data
        # We need to pad to run the segmentation model
        X = np.load(Xfile).transpose(2, 0, 1)
        X = np.pad(X, ((0, 0), (1, 1), (1, 1)), "constant", constant_values=0)
        
        # We convert the mask to a single channel:
        # 0: no cloud, 1: cloud, 2: shadow
        y = np.load(yfile).transpose(2, 0, 1)
        y = y[0] * 0 + y[1] * 1 + y[2] * 2
        y = np.pad(y, ((1, 1), (1, 1)), "constant")

        # From numpy to torch
        X = torch.from_numpy(X).type(torch.float)
        y = torch.from_numpy(y).type(torch.long)
        
        return X, y

# Read netcdf file
root = pathlib.Path("/media/csaybar/2F9A60C90A2CC0FB/IGARSS2023/cloudcatalogue")
dataset = pd.read_csv(root / "classification_tags.csv")
cloudcatalogue_files = CloudDataset(cloudcatalogue_find(root, dataset))