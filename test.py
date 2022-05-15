import torch
from models import *

model = MobileNetV2withConvInsert3_bn()
model.load_state_dict(state_dict=torch.load("./model3_imagenet_cpu.pth"))
