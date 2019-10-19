import os
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def make_datasets(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if is_image_file(target):
            images.append(d)
    return images


def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    item = os.path.join(root, fname)
                    images.append(item)
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with Image.open(path) as img:
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class TargetDataset(data.Dataset):
    def __init__(self, root, transform1=None, transform2=None, loader=default_loader):
        imgs = make_datasets(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n" 
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform1 = transform1
        self.transform2 = transform2
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.imgs[index]
        img1 = self.loader(path)
        img2 = img1
        if self.transform1 is not None:
            img1 = self.transform1(img1)
        if self.transform2 is not None:
            img2 = self.transform2(img2)
        return img1, img2

    def __len__(self):
        return len(self.imgs)
