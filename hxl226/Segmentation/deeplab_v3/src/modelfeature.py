import torch
import torch.nn as nn
from resnet_backbone import resnet50, resnet101
from mobilenet_backbone import mobilenet_v3_large

print(mobilenet_v3_large().feature)

