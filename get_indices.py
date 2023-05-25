import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import timm
import torch

from dataloader_cloudcatalogue import cloudcatalogue_files
from dataloader_cloudsen12 import testing_data, training_data, validation_data
from dataloader_kappaset import kappaset_files
from utils import cloudcatalogue_table, cloudsen12_table, kappaset_table

# Hardness index model
HImodel = timm.create_model("resnet10t", pretrained=True, num_classes=1, in_chans=13)
HImodel.load_state_dict(torch.load("weights/resnet10t.pt"))
HImodel.eval()
HImodel.cuda()


# Trustworthiness index model
TImodel = smp.Unet(encoder_name="mobilenet_v2", in_channels=13, classes=4)
TImodel.load_state_dict(torch.load("weights/UNetMobV2.pt"))
TImodel.eval()
TImodel.cuda()


# CloudSEN12 table with TI and HI indices
train_cloudsen12_db = cloudsen12_table(training_data, HImodel, TImodel)
val_cloudsen12_db = cloudsen12_table(validation_data, HImodel, TImodel)
test_cloudsen12_db = cloudsen12_table(testing_data, HImodel, TImodel)
cloudsen12_db = pd.concat(
    [train_cloudsen12_db, val_cloudsen12_db, test_cloudsen12_db], axis=0
)
cloudsen12_db.to_csv("results/cloudsen12_indices.csv", index=False)

# Kappaset table with TI and HI indices
train_kappaset_db = kappaset_table(kappaset_files, HImodel, TImodel)
train_kappaset_db.to_csv("results/kappaset_indices.csv", index=False)

# CloudCatalogue table with TI and HI indices
train_cloudcatalogue_db = cloudcatalogue_table(cloudcatalogue_files, HImodel, TImodel)
train_cloudcatalogue_db.to_csv("results/cloudcatalogue_indices.csv", index=False)
