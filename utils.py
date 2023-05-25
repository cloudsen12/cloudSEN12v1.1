import pathlib

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch

import dataloader_cloudcatalogue
import dataloader_cloudsen12
import dataloader_kappaset


def cloudsen12_table(dataloader, HImodel, TImodel):
    ctrain_hi = list()
    ctrain_ti = list()

    for item in range(len(dataloader)):
        print(item)
        X, y = dataloader[item]

        # Convert from numpy to torch
        X = X.cuda()[None, ...]

        # Get HI and TI indices
        with torch.no_grad():
            hi_index = torch.sigmoid(HImodel(X))
            yhat = torch.argmax(TImodel(X), dim=1)

        # From torch to numpy
        yhat = yhat.detach().cpu().numpy().squeeze().flatten()
        y = y.numpy().flatten()

        # Estimate the f2-score using sklearn
        ti_index = sklearn.metrics.fbeta_score(y, yhat, beta=2.0, average="macro")

        # Append to lists
        ctrain_hi.append(hi_index)
        ctrain_ti.append(ti_index)

    # Convert to float
    cloudcatalogue_hi = [float(item) for item in ctrain_hi]
    cloudcatalogue_ti = [float(item) for item in ctrain_ti]

    # Save results as a table
    return pd.DataFrame({"HI": cloudcatalogue_hi, "TI": cloudcatalogue_ti})


def kappaset_table(dataloader, HImodel, TImodel):
    ctrain_hi = list()
    ctrain_ti = list()

    for item in range(len(dataloader)):
        print(item)
        X, y = dataloader[item]

        # Convert from numpy to torch
        X = X.cuda()[None, ...]

        # Get HI and TI indices
        with torch.no_grad():
            hi_index = torch.sigmoid(HImodel(X))
            yhat = torch.argmax(TImodel(X), dim=1)

        # From torch to numpy
        yhat = yhat.detach().cpu().numpy().squeeze().flatten()
        y = y.numpy().flatten()

        # If the class is 4, we have to remove it
        # class 4 is invalid in the kappaset dataset:
        # (0): UNDEFINED & (5): MISSING
        yhat = yhat[y != 4]
        y = y[y != 4]

        # Estimate the f2-score using sklearn
        ti_index = sklearn.metrics.fbeta_score(y, yhat, beta=2.0, average="macro")

        # Append to lists
        ctrain_hi.append(hi_index)
        ctrain_ti.append(ti_index)

    # Convert to float
    cloudcatalogue_hi = [float(item) for item in ctrain_hi]
    cloudcatalogue_ti = [float(item) for item in ctrain_ti]

    # Save results as a table
    return pd.DataFrame({"HI": cloudcatalogue_hi, "TI": cloudcatalogue_ti})


def cloudcatalogue_table(dataloader, HImodel, TImodel):
    ctrain_hi = list()
    ctrain_ti = list()

    for item in range(len(dataloader)):
        print(item)
        X, y = dataloader[item]

        # Convert from numpy to torch
        X = X.cuda()[None, ...]

        # Get HI and TI indices
        with torch.no_grad():
            hi_index = torch.sigmoid(HImodel(X))
            yhat = torch.argmax(TImodel(X), dim=1)

            # We have to adapt the labels because the cloudcatalogue dataset
            # does not have thin clouds
            yhat[yhat == 2] = 1
            yhat[yhat == 3] = 2

        # From torch to numpy
        yhat = yhat.detach().cpu().numpy().squeeze().flatten()
        y = y.numpy().flatten()

        # Estimate the f2-score using sklearn
        ti_index = sklearn.metrics.fbeta_score(y, yhat, beta=2.0, average="macro")

        # Append to lists
        ctrain_hi.append(hi_index)
        ctrain_ti.append(ti_index)

    # Convert to float
    cloudcatalogue_hi = [float(item) for item in ctrain_hi]
    cloudcatalogue_ti = [float(item) for item in ctrain_ti]

    # Save results as a table
    return pd.DataFrame({"HI": cloudcatalogue_hi, "TI": cloudcatalogue_ti})


def KappaSetViz(root, TImodel):

    # Select image patches detected as potential errors by P1error
    kappaset = pd.read_csv("results/kappaset_indices.csv")
    kappaset.sort_values(by=["TI", "HI"], inplace=True)
    kappaset.reset_index(inplace=True)

    p_errors = np.where((kappaset.TI < 0.50) & (kappaset.HI >= 0.50))[0]
    kappaset_perrors = kappaset.iloc[p_errors]
    kappaset_perrors.reset_index(inplace=True)

    all_filenames = dataloader_kappaset.kappaset_find(root)
    p_errors_files = [all_filenames[i] for i in p_errors]

    # Create a Dataset
    kappaset_data = dataloader_kappaset.CloudDataset(p_errors_files)

    # Create the cmap
    cmap = matplotlib.colors.ListedColormap(["yellow", "green", "red", "blue"])

    # Create a folder to save the figures
    pathlib.Path("results/kappaset").mkdir(parents=True, exist_ok=True)

    # Create a viz for each potential error
    for index, item in enumerate(p_errors_files):
        print(index)

        # Load the image and the mask
        X, y = kappaset_data[index]
        name = item.stem

        # Predict the mask
        yhat = TImodel(torch.unsqueeze(X, 0).cuda())
        yhat = torch.argmax(yhat, dim=1).squeeze().cpu().numpy().astype(np.float32)

        # Artifact because idk how to use matplotlib
        # in R is quite more simple
        y[0, 0] = 1
        y[-1, -1] = 2
        y[-1, 0] = 3
        y[0, -1] = 4

        yhat[0, 0] = 1
        yhat[-1, -1] = 2
        yhat[-1, 0] = 3
        yhat[0, -1] = 4

        # Create a RGB image
        rgb = X[[3, 2, 1]].cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb * 255).astype(np.uint8)
        rgb = np.concatenate(
            [cv2.equalizeHist(band)[..., None] for band in cv2.split(rgb)], axis=-1
        )

        y = y.cpu().numpy().astype(np.float32)

        # Remove the class (0) CLEAR
        y[y == 0] = np.nan
        yhat[yhat == 0] = np.nan

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(rgb)
        ax[0].set_title("RGB")
        ax[1].imshow(rgb)
        ax[1].imshow(y, cmap=cmap)
        ax[1].set_title("Ground Truth")
        ax[2].imshow(rgb)
        ax[2].imshow(yhat, cmap=cmap)
        ax[2].set_title("CloudSEN12 Model Prediction")

        # Legend
        names = ["thick cloud", "thin cloud", "shadow", "missing & undefined"]
        handles = [
            matplotlib.patches.Patch(color=cmap(i), label=names[i]) for i in range(4)
        ]

        # add legend in the bottom
        fig.legend(handles=handles, loc="lower center", ncol=4)

        # main title
        fig.suptitle(
            f"{name} TI: {kappaset_perrors.TI[index]:.3f} - HI: {kappaset_perrors.HI[index]:.3f}"
        )

        plt.savefig("results/kappaset/%04d.png" % index, bbox_inches="tight")


def CloudCatalogueViz(root, TImodel):

    # Select image patches detected as potential errors by P1error
    cloudcatalogue = pd.read_csv("results/cloudcatalogue_indices.csv")
    cloudcatalogue.sort_values(by=["TI", "HI"], inplace=True)
    cloudcatalogue.reset_index(inplace=True)

    p_errors = np.where((cloudcatalogue.TI < 0.50) & (cloudcatalogue.HI >= 0.50))[0]
    cloudcatalogue_perrors = cloudcatalogue.iloc[p_errors]
    cloudcatalogue_perrors.reset_index(inplace=True)

    metadata = pd.read_csv("data/classification_tags.csv")
    all_filenames = dataloader_cloudcatalogue.cloudcatalogue_find(root, metadata)
    p_errors_files = [
        [all_filenames[0][i] for i in p_errors],
        [all_filenames[1][i] for i in p_errors],
    ]

    # Create a Dataset
    cloudcatalogue_data = dataloader_cloudcatalogue.CloudDataset(p_errors_files)

    # Create the cmap
    cmap = matplotlib.colors.ListedColormap(["yellow", "green", "red"])

    # Create a folder to save the figures
    pathlib.Path("results/cloudcatalogue").mkdir(parents=True, exist_ok=True)

    # Create a viz for each potential error
    for index, item in enumerate(p_errors_files[0]):

        print(index)

        # Load the image and the mask
        X, y = cloudcatalogue_data[index]
        name = item.stem

        # Cloud shadows have the code 3 in CloudSEN12 label protocol
        y[y == 2] = 3

        # Predict the mask
        yhat = TImodel(torch.unsqueeze(X, 0).cuda())
        yhat = torch.argmax(yhat, dim=1).squeeze().cpu().numpy().astype(np.float32)

        # Artifact because idk how to use matplotlib
        # in R is quite more simple
        y[0, 0] = 1
        y[-1, -1] = 2
        y[-1, 0] = 3

        yhat[0, 0] = 1
        yhat[-1, -1] = 2
        yhat[-1, 0] = 3

        # Create a RGB image
        rgb = X[[3, 2, 1]].cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb * 255).astype(np.uint8)
        rgb = np.concatenate(
            [cv2.equalizeHist(band)[..., None] for band in cv2.split(rgb)], axis=-1
        )

        y = y.cpu().numpy().astype(np.float32)

        # Remove the class (0) CLEAR
        y[y == 0] = np.nan
        yhat[yhat == 0] = np.nan

        # Plot the image
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(rgb)
        ax[0].set_title("RGB")
        ax[1].imshow(rgb)
        ax[1].imshow(y, cmap=cmap)
        ax[1].set_title("Ground Truth")
        ax[2].imshow(rgb)
        ax[2].imshow(yhat, cmap=cmap)
        ax[2].set_title("CloudSEN12 Model Prediction")

        # Legend
        names = ["thick cloud", "thin cloud", "shadow"]
        handles = [
            matplotlib.patches.Patch(color=cmap(i), label=names[i]) for i in range(3)
        ]

        # add legend in the bottom
        fig.legend(handles=handles, loc="lower center", ncol=4)

        # main title
        fig.suptitle(
            f"{name} TI: {cloudcatalogue_perrors.TI[index]:.3f} - HI: {cloudcatalogue_perrors.HI[index]:.3f}"
        )
        plt.savefig("results/cloudcatalogue/%04d.png" % index, bbox_inches="tight")
        plt.close(fig)


def CloudSEN12Viz(root, TImodel):
    # Select image patches detected as potential errors by P1error
    cloudsen12 = pd.read_csv("cloudsen12_indices.csv")
    p_errors = np.where((cloudsen12.TI < 0.50) & (cloudsen12.HI >= 0.50))[0]
    cloudcatalogue_perrors = cloudsen12.iloc[p_errors]

    # CloudSEN12 metadata
    f1 = pd.read_csv(pathlib.Path(root) / "high_train" / "metadata.csv")
    f2 = pd.read_csv(pathlib.Path(root) / "high_val" / "metadata.csv")
    f3 = pd.read_csv(pathlib.Path(root) / "high_test" / "metadata.csv")
    f = pd.concat([f1, f2, f3])
    f.reset_index(drop=True, inplace=True)
    PKcloudSEN12 = (f.roi_id + "__" + f.s2_id_gee).iloc[p_errors]
    indices = pd.Series(PKcloudSEN12.index)

    # CloudSEN12 dataloader
    training_data = dataloader_cloudsen12.CloudDataset(
        root=root, type="train", model="seg"
    )
    validation_data = dataloader_cloudsen12.CloudDataset(
        root=root, type="val", model="seg"
    )
    testing_data = dataloader_cloudsen12.CloudDataset(
        root=root, type="test", model="seg"
    )

    # Create the cmap
    cmap = matplotlib.colors.ListedColormap(["yellow", "green", "red"])

    # Create a folder to save the figures
    pathlib.Path("results/cloudsen12").mkdir(parents=True, exist_ok=True)

    PKcloudSEN12.reset_index(drop=True, inplace=True)
    cloudcatalogue_perrors.reset_index(drop=True, inplace=True)

    cloudcatalogue_perrors = cloudcatalogue_perrors.sort_values(by=["TI", "HI"])
    PKcloudSEN12 = PKcloudSEN12.iloc[cloudcatalogue_perrors.index]
    indices = indices.iloc[cloudcatalogue_perrors.index]

    cloudcatalogue_perrors.reset_index(drop=True, inplace=True)
    PKcloudSEN12.reset_index(drop=True, inplace=True)
    indices.reset_index(drop=True, inplace=True)

    import re

    [i for i, x in enumerate(list(PKcloudSEN12)) if re.search("ROI_0007", x)]

    # Create a viz for each potential error
    for index in range(len(PKcloudSEN12)):
        item = indices[index]
        if item < len(training_data):
            X, y = training_data[item]
        elif len(training_data) <= item < (len(training_data) + len(validation_data)):
            X, y = validation_data[item - len(training_data)]
        elif (
            (len(training_data) + len(validation_data))
            <= item
            < (len(training_data) + len(validation_data) + len(testing_data))
        ):
            X, y = testing_data[item - len(training_data) - len(validation_data)]
        name = PKcloudSEN12[index]

        # Predict the mask
        yhat = TImodel(torch.unsqueeze(X, 0).cuda())
        yhat = torch.argmax(yhat, dim=1).squeeze().cpu().numpy().astype(np.float32)

        # Artifact because idk how to use matplotlib
        # in R is quite more simple
        y[0, 0] = 1
        y[-1, -1] = 2
        y[-1, 0] = 3

        yhat[0, 0] = 1
        yhat[-1, -1] = 2
        yhat[-1, 0] = 3

        # Create a RGB image
        rgb = X[[3, 2, 1]].cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb * 255).astype(np.uint8)
        rgb = np.concatenate(
            [cv2.equalizeHist(band)[..., None] for band in cv2.split(rgb)], axis=-1
        )

        y = y.cpu().numpy().astype(np.float32)

        # Remove the class (0) CLEAR
        y[y == 0] = np.nan
        yhat[yhat == 0] = np.nan

        # Plot the image
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(rgb)
        ax[0].set_title("RGB")
        ax[1].imshow(rgb)
        ax[1].imshow(y, cmap=cmap)
        ax[1].set_title("Ground Truth")
        ax[2].imshow(rgb)
        ax[2].imshow(yhat, cmap=cmap)
        ax[2].set_title("CloudSEN12 Model Prediction")

        # Legend
        names = ["thick cloud", "thin cloud", "shadow"]
        handles = [
            matplotlib.patches.Patch(color=cmap(i), label=names[i]) for i in range(3)
        ]

        # add legend in the bottom
        fig.legend(handles=handles, loc="lower center", ncol=4)

        # main title
        fig.suptitle(
            f"{name} TI: {cloudcatalogue_perrors.TI[index]:.3f} - HI: {cloudcatalogue_perrors.HI[index]:.3f}"
        )
        plt.savefig("results/cloudsen12/%04d.png" % index, bbox_inches="tight")
        plt.close(fig)
