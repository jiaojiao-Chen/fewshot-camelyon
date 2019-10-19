import glob
import os

import cv2
import numpy as np

import utils as utils
from wsi_ops import PatchExtractor
from wsi_ops import WSIOps


def extract_positive_patches_from_tumor_wsi(wsi_ops, patch_extractor, patch_index, augmentation=False):
    wsi_paths = glob.glob(os.path.join(utils.TUMOR_WSI_PATH, '*.tif'))
    wsi_paths.sort()
    mask_paths = glob.glob(os.path.join(utils.TUMOR_MASK_PATH, '*.tif'))
    mask_paths.sort()

    image_mask_pair = zip(wsi_paths, mask_paths)
    image_mask_pair = list(image_mask_pair)

    patch_save_dir = '/home/jiaojiao/patch/299/Tumor/'
    patch_prefix = utils.PATCH_AUG_TUMOR_PREFIX if augmentation else utils.PATCH_TUMOR_PREFIX

    for image_path, mask_path in image_mask_pair:
        print('extract_positive_patches_from_tumor_wsi(): %s' % utils.get_filename_from_path(image_path))
        wsi_image, rgb_image, wsi_mask, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(image_path, mask_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        bounding_boxes = wsi_ops.find_roi_bbox_tumor_gt_mask(np.array(tumor_gt_mask))

        patch_index = patch_extractor.extract_positive_patches_from_tumor_region(wsi_image, wsi_mask,
                                                                                 level_used, bounding_boxes,
                                                                                 patch_save_dir, patch_prefix,
                                                                                 patch_index)
        print('Positive patch count: %d' % (patch_index - utils.PATCH_INDEX_POSITIVE))
        wsi_image.close()
    return patch_index


def extract_negative_patches_from_tumor_wsi(wsi_ops, patch_extractor, patch_index, augmentation=False):
    wsi_paths = glob.glob(os.path.join(utils.TUMOR_WSI_PATH, '*.tif'))
    wsi_paths.sort()
    mask_paths = glob.glob(os.path.join(utils.TUMOR_MASK_PATH, '*.tif'))
    mask_paths.sort()

    image_mask_pair = zip(wsi_paths, mask_paths)
    image_mask_pair = list(image_mask_pair)
    # image_mask_pair = image_mask_pair[67:68]

    patch_save_dir = '/home/jiaojiao/patch/299/Normal/'
    patch_prefix = utils.PATCH_AUG_NORMAL_PREFIX if augmentation else utils.PATCH_NORMAL_PREFIX
    for image_path, mask_path in image_mask_pair:
        print('extract_negative_patches_from_tumor_wsi(): %s' % utils.get_filename_from_path(image_path))
        wsi_image, rgb_image, wsi_mask, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(image_path, mask_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        bounding_boxes, _, image_open = wsi_ops.find_roi_bbox(np.array(rgb_image))

        patch_index = patch_extractor.extract_negative_patches_from_tumor_wsi(wsi_image, wsi_mask,
                                                                              image_open, level_used,
                                                                              bounding_boxes, patch_save_dir,
                                                                              patch_prefix,
                                                                              patch_index)
        print('Negative patches count: %d' % (patch_index - utils.PATCH_INDEX_NEGATIVE))

        wsi_image.close()
        wsi_mask.close()

    return patch_index

def extract_negative_patches_from_normal_wsi(wsi_ops, patch_extractor, patch_index, augmentation=False):
    """
    Extracted up to Normal_060.

    :param wsi_ops:
    :param patch_extractor:
    :param patch_index:
    :param augmentation:
    :return:
    """
    wsi_paths = glob.glob(os.path.join(utils.NORMAL_WSI_PATH, '*.tif'))
    wsi_paths.sort()

    #wsi_paths = wsi_paths[110:]
    #wsi_paths = wsi_paths[0:60]
    patch_save_dir = '/home/jiaojiao/patch/299/Normal/'
    patch_prefix = utils.PATCH_AUG_NORMAL_PREFIX if augmentation else utils.PATCH_NORMAL_PREFIX
    for image_path in wsi_paths:
        print('extract_negative_patches_from_normal_wsi(): %s' % utils.get_filename_from_path(image_path))
        wsi_image, rgb_image, level_used = wsi_ops.read_wsi_normal(image_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        bounding_boxes, _, image_open = wsi_ops.find_roi_bbox(np.array(rgb_image))

        patch_index = patch_extractor.extract_negative_patches_from_normal_wsi(wsi_image, image_open,
                                                                               level_used,
                                                                               bounding_boxes,
                                                                               patch_save_dir, patch_prefix,
                                                                               patch_index)
        print('Negative patches count: %d' % (patch_index - utils.PATCH_INDEX_NEGATIVE))

        wsi_image.close()

    return patch_index

def extract_patches_from_heatmap_false_region_tumor(wsi_ops, patch_extractor, patch_index_pos, patch_index_neg, augmentation=False):
    tumor_heatmap_prob_paths = glob.glob(os.path.join('/home/jiaojiao/PycharmProjects/camelyon16/heatmap_prob/false', '*heatmap*umor*.png'))
    tumor_heatmap_prob_paths.sort()
    print(len(tumor_heatmap_prob_paths))

    wsi_paths = glob.glob(os.path.join(utils.TUMOR_WSI_PATH+'/true', '*.tif'))
    wsi_paths.sort()


    mask_paths = glob.glob(os.path.join(utils.TUMOR_MASK_PATH+'/true', '*.tif'))
    mask_paths.sort()

    assert len(tumor_heatmap_prob_paths) == len(wsi_paths), 'Some heatmaps are missing!'

    image_mask_heatmap_tuple = zip(wsi_paths, mask_paths, tumor_heatmap_prob_paths)
    image_mask_heatmap_tuple = list(image_mask_heatmap_tuple)

    #patch_save_dir_pos = utils.PATCHES_TRAIN_AUG_EXCLUDE_MIRROR_WSI_POSITIVE_PATH if augmentation else utils.PATCHES_TRAIN_POSITIVE_PATH
    patch_save_dir_pos = '/home/jiaojiao/patch/299/tumor/'
    patch_prefix_pos = utils.PATCH_AUG_TUMOR_PREFIX if augmentation else utils.PATCH_TUMOR_PREFIX
    #patch_save_dir_neg = utils.PATCHES_TRAIN_AUG_EXCLUDE_MIRROR_WSI_NEGATIVE_PATH if augmentation else utils.PATCHES_TRAIN_NEGATIVE_PATH
    patch_save_dir_neg = '/home/jiaojiao/patch/299/normal/'
    patch_prefix_neg = utils.PATCH_AUG_NORMAL_PREFIX if augmentation else utils.PATCH_NORMAL_PREFIX
    not_0_255_cnt = 0
    for image_path, mask_path, heatmap_prob_path in image_mask_heatmap_tuple:
        print('extract_patches_from_heatmap_false_region_normal(): %s, %s, %s' %
              (utils.get_filename_from_path(image_path), utils.get_filename_from_path(mask_path),
               utils.get_filename_from_path(heatmap_prob_path)))

        wsi_image, rgb_image, wsi_mask, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(image_path, mask_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        bounding_boxes, _, image_open = wsi_ops.find_roi_bbox(np.array(rgb_image))

        heatmap_prob = cv2.imread(heatmap_prob_path)
        heatmap_prob = heatmap_prob[:, :, :1]
        heatmap_prob = np.reshape(heatmap_prob, (heatmap_prob.shape[0], heatmap_prob.shape[1]))
        heatmap_prob = np.array(heatmap_prob, dtype=np.float32)
        heatmap_prob /= 255

        patch_index_neg, patch_index_pos = patch_extractor.extract_patches_from_heatmap_false_region_tumor(wsi_image,
                                                                                      wsi_mask,
                                                                                      tumor_gt_mask,
                                                                                      image_open,
                                                                                      heatmap_prob,
                                                                                      level_used, bounding_boxes,
                                                                                      patch_save_dir_pos,
                                                                                      patch_save_dir_neg,
                                                                                      patch_prefix_pos,
                                                                                      patch_prefix_neg,
                                                                                      patch_index_pos,
                                                                                      patch_index_neg)
        print('negative patch count: %d' % (patch_index_neg - utils.PATCH_INDEX_NEGATIVE))
        print('positive patch count: %d' % (patch_index_pos - utils.PATCH_INDEX_POSITIVE))

        wsi_image.close()
        wsi_mask.close()

    # print('not_0_255_cnt: %d' % not_0_255_cnt)
    return patch_index_neg, patch_index_pos


def extract_patches_from_heatmap_false_region_normal(wsi_ops, patch_extractor, patch_index, augmentation=False):
    normal_heatmap_prob_paths = glob.glob(os.path.join('/home/jiaojiao/PycharmProjects/camelyon16/heatmap_prob/false', 'heatmap*ormal*.png'))
    normal_heatmap_prob_paths.sort()
    #print(normal_heatmap_prob_paths)
    wsi_paths = glob.glob(os.path.join(utils.NORMAL_WSI_PATH+'/true', '*.tif'))
    wsi_paths.sort()
    assert len(normal_heatmap_prob_paths) == len(wsi_paths), 'Some heatmaps are missing!'

    image_heatmap_tuple = zip(wsi_paths, normal_heatmap_prob_paths)
    image_heatmap_tuple = list(image_heatmap_tuple)
    # image_mask_pair = image_mask_pair[67:68]

    patch_save_dir_neg = '/home/jiaojiao/patch/299/normal_from_normal/'
    patch_prefix_neg = utils.PATCH_AUG_NORMAL_PREFIX if augmentation else utils.PATCH_NORMAL_PREFIX
    for image_path, heatmap_prob_path in image_heatmap_tuple:
        print('extract_patches_from_heatmap_false_region_normal(): %s, %s' % (utils.get_filename_from_path(image_path)
                                                                              , utils.get_filename_from_path(
            heatmap_prob_path)))

        wsi_image, rgb_image, level_used = wsi_ops.read_wsi_normal(image_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        bounding_boxes, _, image_open = wsi_ops.find_roi_bbox(np.array(rgb_image))

        heatmap_prob = cv2.imread(heatmap_prob_path)
        heatmap_prob = heatmap_prob[:, :, :1]
        heatmap_prob = np.reshape(heatmap_prob, (heatmap_prob.shape[0], heatmap_prob.shape[1]))
        heatmap_prob = np.array(heatmap_prob, dtype=np.float32)
        heatmap_prob /= 255

        patch_index = patch_extractor.extract_patches_from_heatmap_false_region_normal(wsi_image,
                                                                                       image_open,
                                                                                       heatmap_prob,
                                                                                       level_used, bounding_boxes,
                                                                                       patch_save_dir_neg,
                                                                                       patch_prefix_neg,
                                                                                       patch_index)
        print('patch count: %d' % (patch_index - utils.PATCH_INDEX_NEGATIVE))

        wsi_image.close()

    return patch_index


def extract_patches(ops, pe):
    patch_index_positive = utils.PATCH_INDEX_POSITIVE
    patch_index_negative = utils.PATCH_INDEX_NEGATIVE
    #patch_index_negative = extract_negative_patches_from_tumor_wsi(ops, pe, patch_index_negative)
    #extract_negative_patches_from_normal_wsi(ops, pe, patch_index_negative)
    extract_positive_patches_from_tumor_wsi(ops, pe, patch_index_positive)


def extract_patches_augmented(ops, pe):
    patch_index_positive = utils.PATCH_INDEX_POSITIVE
    patch_index_negative = utils.PATCH_INDEX_NEGATIVE
    # index - 500000
    # index - 700000 -> remove wrong false positives
    #patch_index_negative, patch_index_pos = extract_patches_from_heatmap_false_region_tumor(ops, pe, patch_index_positive, patch_index_negative)
    #print(patch_index_negative)
    #print(patch_index_pos)
    # index - 600000
    patch_index_negative = extract_patches_from_heatmap_false_region_normal(ops, pe, patch_index_negative)
    print(patch_index_negative)
    # patch_index_negative = extract_negative_patches_from_tumor_wsi(ops, pe, patch_index_negative, augmentation=True)
    # extract_negative_patches_from_normal_wsi(ops, pe, patch_index_negative, augmentation=True)
    # extract_positive_patches_from_tumor_wsi(ops, pe, patch_index_positive, augmentation=True)


if __name__ == '__main__':
    # extract_patches(WSIOps(), PatchExtractor())
    extract_patches_augmented(WSIOps(), PatchExtractor())

