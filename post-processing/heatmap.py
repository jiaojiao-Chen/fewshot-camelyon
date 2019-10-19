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
# INCEPTION/camelyon/Figure_4.pth
# INCEPTION/finetune/Figure_1.pth
# load the model
model = torchvision.models.inception_v3(pretrained=False, num_classes=2)
model.eval()
model.load_state_dict(torch.load('./INCEPTION/camelyon/Figure_4.pth'))
model.cuda()

normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
    )
preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

path = '/fast/TestSet'

filelist = os.listdir(path)

filelist.sort()

for file in filelist:
    # print(file)
    since = time.time()
    wsi_path = os.path.join(path, file)
    wsi_image = OpenSlide(wsi_path)
    level = 6
    width, height = wsi_image.level_dimensions[level]
    rgb_image_pil = wsi_image.read_region((0, 0), level, (width, height))
    rgb_image = np.array(rgb_image_pil)
    cv2.imwrite('./heatmap/finetune/'+file[0:-4]+'.jpg', rgb_image)

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    open_kernel = np.ones((5, 5), np.uint8)

    # close and open operation
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=close_kernel)
    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel=open_kernel)

    output_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # find the contour of mask
    _, contour, hierarchy = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # line_color = (255, 0, 0)
    # cv2.drawContours(rgb_image, contour, -1, line_color, 2)
    cv2.imwrite('./heatmap/finetune/contour_'+file[0:-4]+'.jpg', rgb_image)
    """"""
    # find the bounding box of mask
    bounding_boxes = [cv2.boundingRect(c) for c in contour]

    # build the array
    mm = np.zeros([height, width])

    mag_factor = pow(2, 6)
    # get the patches
    for bounding_box in bounding_boxes:
        b_x_start = int(bounding_box[0])
        b_y_start = int(bounding_box[1])
        b_x_end = int(bounding_box[0]) + int(bounding_box[2])
        b_y_end = int(bounding_box[1]) + int(bounding_box[3])
        step = 1
        X = np.arange(b_x_start, b_x_end, step)
        Y = np.arange(b_y_start, b_y_end, step)
        for x in X:
            for y in Y:
                if int(mask_open[y, x]) is not 0:
                    patch = wsi_image.read_region((x * mag_factor, y * mag_factor), 0, (299, 299))
                    patch_array = np.array(patch)
                    img = Image.fromarray(patch_array[:, :, :3])
                    img_tensor = preprocess(img)
                    img_variable = Variable(img_tensor.unsqueeze(0).cuda())
                    logit = model(img_variable)
                    h_x = F.softmax(logit).data.squeeze()
                    mm[y, x] = h_x[1]
                    patch.close()

    cv2.imwrite('./heatmap/finetune/black_' + file[0:-4] + '.jpg', mm * 255)

    # mm = mm[::-1]
    plt.imshow(mm, cmap='jet', interpolation='nearest')
    # plt.colorbar()
    # plt.clim(0.00, 1.00)
    # plt.axis([0, rgb_image.shape[1], 0, rgb_image.shape[0]])
    plt.savefig(str(os.path.join('./heatmap/finetune', 'color_'+file[0:-4] + '.jpg')))
    plt.clf()

    nn = mm
    nn[nn < 0.5] = 0
    cv2.imwrite('./heatmap/finetune/clear_' + file[0:-4] + '.jpg', nn * 255)

    time_elapsed = time.time() - since
    print(file[0:-4]+' complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
