import numpy as np
import cv2
import os
from PIL import Image
from openslide import OpenSlide
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
import time


path = '/media/jiaojiao/Seagate Backup Plus Drive/CAMELYON16/TestingData/Testset'

filelist = os.listdir(path)

filelist.sort()

for file in filelist:
    wsi_path = os.path.join(path, file)
    wsi_image = OpenSlide(wsi_path)
    level = 7
    width, height = wsi_image.level_dimensions[level]
    print(width, height)