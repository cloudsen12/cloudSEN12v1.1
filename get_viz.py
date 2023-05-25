import pathlib

import segmentation_models_pytorch as smp
import torch

from utils import KappaSetViz

root = pathlib.Path("/media/csaybar/2F9A60C90A2CC0FB/IGARSS2023")

# Trustworthiness index model
TImodel = smp.Unet(encoder_name="mobilenet_v2", in_channels=13, classes=4)
TImodel.load_state_dict(
    torch.load("weights/UNetMobV2.pt", map_location=torch.device("cpu"))
)
TImodel.eval()
TImodel.cuda()

# Display potential errors - KappaSet
KappaSetViz(root / "kappaset", TImodel)

# Display potential errors - CloudCatalogue
CloudCatalogueViz(root / "cloudcatalogue", TImodel)

# Display potential errors - CloudSEN12
CloudSEN12Viz(root / "cloudsen12/high/", TImodel)
