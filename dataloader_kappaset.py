import xarray as xr
import numpy as np
import pathlib
import torch

def kappaset_find(root):
    """
    Retrieve the paths of NetCDF files within subfolders.

    Args:
        root (pathlib.Path): The root directory containing subfolders.

    Returns:
        list: A list of file paths.

    """
    subfolders = [
        "April", "August", "December", "February", "January", "July", 
        "June", "March", "May", "November", "October", "September", "test"
    ]
    container = []
    for subfolder in subfolders:
        for file in root.glob(f"{subfolder}/*.nc"):
            container.append(file)
    return container

class CloudDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        """
        Cloud dataset class for loading and preprocessing NetCDF files.

        Args:
            files (list): A list of file paths.

        """
        self.kappafiles = files

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.kappafiles)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple: A tuple containing the input and target data.

        """
        kappafile = self.kappafiles[idx]
        data = xr.open_dataset(kappafile)
        
        # CloudSEN12 labels code conversion
        shadow = np.array(data.Label == 2) * 3
        thick_cloud = np.array(data.Label == 4) * 1
        thin_cloud = np.array(data.Label == 3) * 2
        invalid = np.array((data.Label == 5) | (data.Label == 0)) * 4
        mask = shadow + thick_cloud + thin_cloud + invalid
        
        # Apply KappaSet scaling factor
        s2bands = [
            "B01", "B02", "B03", "B04", "B05", "B06","B07", "B08",
            "B8A", "B09", "B10", "B11", "B12"
        ]
        l1c = np.array(data[s2bands].to_array())* 6.5535
        
        # From numpy to torch
        X = torch.from_numpy(l1c).type(torch.float)
        y = torch.from_numpy(mask).type(torch.long)
        
        return X, y
        
# Create DataLoader
root = pathlib.Path("/media/csaybar/2F9A60C90A2CC0FB/IGARSS2023/kappaset")
kappaset_files = CloudDataset(kappaset_find(root))