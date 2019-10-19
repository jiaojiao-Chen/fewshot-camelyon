import os
from openslide import OpenSlide
import cv2
import numpy as np
import random
files = os.listdir('/media/jiaojiao/Seagate Backup Plus Drive/CAMELYON16/TrainingData/pink')
# mask_dir = '/media/jiaojiao/Seagate Backup Plus Drive/CAMELYON16/TrainingData/Ground_Truth/Mask/'
files.sort()
patch_index = 0

for file in files:
    name = file.split('.')[0]
    print(name)
    wsi_path = '/media/jiaojiao/Seagate Backup Plus Drive/CAMELYON16/TrainingData/pink/' + file
    wsi_image = OpenSlide(wsi_path)
    level = 7
    rgb_image_pil = wsi_image.read_region((0, 0), level, wsi_image.level_dimensions[level])
    rgb_image = np.array(rgb_image_pil)

    # get the mask
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([40, 40, 40])
    upper_red = np.array([200, 200, 200])
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    open_kernel = np.ones((5, 5), dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_red, upper_red)

    # close and open operation
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=close_kernel)

    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel=open_kernel)

    # find the contour of mask
    _, contour, hierarchy = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the bounding box of mask
    bounding_boxes = [cv2.boundingRect(c) for c in contour]
    # print(bounding_boxes)
    mag_factor = pow(2, level)

    # get the patches

    for bounding_box in bounding_boxes:
        b_x_start = int(bounding_box[0])
        b_y_start = int(bounding_box[1])
        b_x_end = int(bounding_box[0]) + int(bounding_box[2]) - 1
        b_y_end = int(bounding_box[1]) + int(bounding_box[3]) - 1
        step = 20
        X = np.arange(b_x_start, b_x_end, step)
        Y = np.arange(b_y_start, b_y_end, step)
        for x in X:
            for y in Y:
                if int(mask_open[y, x]) is not 0:
                    r = random.randint(300, 800)
                    patch = wsi_image.read_region((x * mag_factor, y * mag_factor), 0, (r, r))
                    patch_array = np.array(patch)
                    patch_array = cv2.resize(patch_array, (299, 299))
                    patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
                    lower_red = np.array([40, 40, 40])
                    upper_red = np.array([200, 200, 200])
                    mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
                    white_pixel_cnt = cv2.countNonZero(mask_patch)     # 0.2 pink  0.5 purple
                    if white_pixel_cnt < ((299 * 299) * 0.20) and white_pixel_cnt > ((299 * 299) * 0.05):
                        r = random.random()
                        if r < 0.3:
                            patch.save('/home/jiaojiao/patch/One-Shot/rank/' + str(patch_index) + '.png', 'PNG')
                            patch_index += 1
                    if white_pixel_cnt >= ((299 * 299) * 0.20):
                        patch.save('/home/jiaojiao/patch/One-Shot/rank/' + str(patch_index) + '.png', 'PNG')
                        patch_index += 1
                    patch.close()
print(patch_index)
