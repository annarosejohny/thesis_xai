import torch
import torch.nn as nn
from my_arcface import MyArcFace
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import datasets
from torch.utils import data
from torchvision import transforms
from PIL import Image
import glob
import numpy as np
from torchvision.transforms import ToTensor
# use the ImageNet transformation
# model = MyArcFace(num_classes = 204)
# model.eval()

path = '/home/ash/Desktop/thesis_anna/Datasets/GMDB/organized_data/class_split/train/0/0_3524.jpg'
# for f in glob.glob(path):



#     dataset = datasets.ImageFolder(root='/home/ash/Desktop/thesis_anna/Datasets/GMDB/organized_data/class_split/train/')
#     train_loader = torch.utils.data.DataLoader(dataset)
#     print(train_loader)
# # image_size = 112
# img_path = "/home/ash/Desktop/thesis_anna/GestaltMatcher-Arc/data/GestaltMatcherDB/v1.0.3/gmdb_align/1_rot_aligned.jpg"
#
# image = Image.open(img_path).convert("RGB")
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
#
# img_tensor = transform(image)
# model(img_tensor.unsqueeze(0)).backward()


