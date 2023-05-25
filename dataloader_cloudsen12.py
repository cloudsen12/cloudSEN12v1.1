import pandas as pd
import numpy as np
import pathlib
import torch

def load_data(path_root, shape):
    """
    Load data from memory-mapped files.

    Args:
        path_root (pathlib.Path): The root directory path.
        shape (tuple): The shape of the data.

    Returns:
        dict: A dictionary containing the loaded data.

    """
    total = {
        "B1": np.memmap(path_root / "L1C_B1.dat", dtype=np.int16, mode="r", shape=shape),
        "B2": np.memmap(path_root / "L1C_B2.dat", dtype=np.int16, mode="r", shape=shape),
        "B3": np.memmap(path_root / "L1C_B3.dat", dtype=np.int16, mode="r", shape=shape),
        "B4": np.memmap(path_root / "L1C_B4.dat", dtype=np.int16, mode="r", shape=shape),
        "B5": np.memmap(path_root / "L1C_B5.dat", dtype=np.int16, mode="r", shape=shape),
        "B6": np.memmap(path_root / "L1C_B6.dat", dtype=np.int16, mode="r", shape=shape),
        "B7": np.memmap(path_root / "L1C_B7.dat", dtype=np.int16, mode="r", shape=shape),
        "B8": np.memmap(path_root / "L1C_B8.dat", dtype=np.int16, mode="r", shape=shape),
        "B8A": np.memmap(path_root / "L1C_B8A.dat", dtype=np.int16, mode="r", shape=shape),
        "B9": np.memmap(path_root / "L1C_B9.dat", dtype=np.int16, mode="r", shape=shape),
        "B10": np.memmap(path_root / "L1C_B10.dat", dtype=np.int16, mode="r", shape=shape),
        "B11": np.memmap(path_root / "L1C_B11.dat", dtype=np.int16, mode="r", shape=shape),
        "B12": np.memmap(path_root / "L1C_B12.dat", dtype=np.int16, mode="r", shape=shape),
        "LABEL": np.memmap(path_root / "LABEL_manual_hq.dat", dtype=np.int8, mode="r", shape=shape)
    }
    return total

class CloudDataset(torch.utils.data.Dataset):
    def __init__(self, root, type, model="reg"):
        """
        Cloud dataset class for loading and preprocessing data.

        Args:
            root (str): The root directory path.
            type (str): The type of dataset ("train", "val", or "test").
            model (str, optional): The model type ("reg" for regression or other for classification). Defaults to "reg".

        """
        self.root = pathlib.Path(root)
        self.type = type
        self.model = model
        if type == "train":
            self.X = load_data(self.root / "high_train", (8490, 512, 512))
            self.y = pd.read_csv(self.root / "high_train" / "metadata.csv")
        if type == "val":
            self.X = load_data(self.root / "high_val", (535, 512, 512))
            self.y = pd.read_csv(self.root / "high_val" / "metadata.csv")
        if type == "test":
            self.X = load_data(self.root / "high_test", (975, 512, 512))
            self.y = pd.read_csv(self.root / "high_test" / "metadata.csv")

        if self.model == "reg":
            self.y = np.array(self.y.difficulty)
        else:
            self.y = self.X["LABEL"]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return self.X["B2"].shape[0]

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (int): The index of the sample.

        Returns:
            tuple: A tuple containing the input and target data.

        """
        X = (
            self.X["B1"][index],
            self.X["B2"][index],
            self.X["B3"][index],
            self.X["B4"][index],
            self.X["B5"][index],
            self.X["B6"][index],
            self.X["B7"][index],
            self.X["B8"][index],
            self.X["B8A"][index],
            self.X["B9"][index],
            self.X["B10"][index],
            self.X["B11"][index],
            self.X["B12"][index]
        )

        # Concatenate the bands
        X = np.stack(X, axis=0)/10000
        
        X = torch.from_numpy(X).type(torch.float)
        if self.model == "reg":
            # Convert to binary classification, 0 for easy and 1 for hard
            y = float(float(self.y[index]) >= 3)
        else:
            y = torch.from_numpy(self.y[index]).type(torch.long)
        return X, y

# Create dataloader
training_data = CloudDataset(root="/data3/cloudsen12_high/", type="train")
validation_data = CloudDataset(root="/data3/cloudsen12_high/", type="val")
testing_data = CloudDataset(root="/data3/cloudsen12_high/", type="test")

