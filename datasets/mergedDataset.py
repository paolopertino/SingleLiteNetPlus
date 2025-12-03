import os
import random

import albumentations as A
import cv2
import numpy as np
import torch

from skimage import io
from utils import artifact_filter

from datasets.utils.transforms import (
    RandomBilateralBlur,
    RandomGaussianBlur,
    augment_hsv,
    letterbox,
    letterbox_for_img,
    random_perspective,
    resize_and_pad_label,
)


class Dataset(torch.utils.data.Dataset):
    """Dataset is a PyTorch dataset class for loading images and their corresponding masks given their folder paths.

    :param images_folder: The folder containing the input images.
    :param annotations_folder: The folder containing the target masks.
    :param hyp: A dictionary containing hyperparameters.
    :param model_name: The name of the model being used.
    :param split: The dataset split (train/val).
    :param lidar_img1: The path to the first LiDAR image.
    :param lidar_img2: The path to the second LiDAR image.
    :param lidar_img3: The path to the third LiDAR image.
    :param idx_remapping: A dictionary for remapping label indices when loading the target masks.
    :param img_resize: Desired image dimension (used for inference)
    """

    def __init__(
        self,
        images_folder: str,
        annotations_folder: str,
        hyp,
        model_name: str,
        split: str = "",
        lidar_img1: str = None,
        lidar_img2: str = None,
        lidar_img3: str = None,
        idx_remapping: dict = None,
        img_resize: int = None,
    ):
        super().__init__()

        self.split = split
        self.is_train = split == "train"
        self.model_name = model_name
        self.idx_remapping = idx_remapping

        self.is_lidar = False
        if lidar_img1 is not None and lidar_img2 is not None and lidar_img3 is not None:
            self.is_lidar = True
            self.lidar_images_path1 = os.path.join(images_folder, split, lidar_img1)
            self.lidar_images_path2 = os.path.join(images_folder, split, lidar_img2)
            self.lidar_images_path3 = os.path.join(images_folder, split, lidar_img3)
            self.names = os.listdir(self.lidar_images_path1)
        else:
            self.images_path = os.path.join(images_folder, split)
            self.names = os.listdir(self.images_path)

        if self.split != "":
            self.annotations_folder = os.path.join(annotations_folder, split)

            # Transformations Hyperparameters
            self.degrees = hyp["degrees"]
            self.translate = hyp["translate"]
            self.scale = hyp["scale"]
            self.shear = hyp["shear"]
            self.hgain = hyp["hgain"]
            self.sgain = hyp["sgain"]
            self.vgain = hyp["vgain"]
            self.Random_Crop = A.RandomCrop(
                width=hyp["width_crop"], height=hyp["height_crop"]
            )
            self.resize_largest_side = hyp["resize_largest_side"]

            self.prob_perspective = hyp["prob_perspective"]
            self.prob_flip = hyp["prob_flip"]
            self.prob_hsv = hyp["prob_hsv"]
            self.prob_bilateral = hyp["prob_bilateral"]
            self.prob_gaussian = hyp["prob_gaussian"]
            self.prob_crop = hyp["prob_crop"]
        else:
            self.img_resize = img_resize

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.names)

    def __getitem__(self, idx: int):
        """Retrieves an item from the dataset at the specified index.

        :param idx: The index of the image and mask to retrieve.
        """

        def read_greyscale_image(img):
            if img.ndim == 3 and img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return img

        if self.split != "":
            label = cv2.imread(
                os.path.join(self.annotations_folder, self.names[idx]).replace(
                    ".jpg", ".png"
                ),
                cv2.IMREAD_UNCHANGED,
            )

        image_name = self.names[idx]
        if self.is_lidar:
            image_name1 = os.path.join(self.lidar_images_path1, image_name)
            image_name2 = os.path.join(self.lidar_images_path2, image_name)
            image_name3 = os.path.join(self.lidar_images_path3, image_name)

            # Image loading
            img1 = read_greyscale_image(io.imread(image_name1))
            img2 = read_greyscale_image(io.imread(image_name2))
            img3 = read_greyscale_image(io.imread(image_name3))

            # Filtering
            img1 = np.uint8(np.clip(artifact_filter(img1), 0, 255))
            img2 = np.uint8(np.clip(artifact_filter(img2), 0, 255))
            img3 = np.uint8(np.clip(artifact_filter(img3), 0, 255))

            # Central cropping
            img1 = img1[:, 256:768]
            img2 = img2[:, 256:768]
            img3 = img3[:, 256:768]
            if self.split != "":
                label = label[:, 256:768]

            # Stacking
            rgb_image = np.stack([img1, img2, img3], axis=-1).astype(np.uint8)
            image = np.ascontiguousarray(rgb_image[..., ::-1])  # Convert RGB to BGR
        else:
            image_path = os.path.join(self.images_path, image_name)
            image = cv2.imread(image_path)

        if self.is_train:
            if random.random() < self.prob_perspective:
                combination = (image, label)
                (image, label) = random_perspective(
                    combination=combination,
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear,
                )
            if random.random() < self.prob_hsv:
                augment_hsv(image, self.hgain, self.sgain, self.vgain)
            if random.random() < self.prob_flip:
                image = np.fliplr(image)
                label = np.fliplr(label)
            if random.random() < self.prob_bilateral:
                image = RandomBilateralBlur(image)
            if random.random() < self.prob_gaussian:
                image = RandomGaussianBlur(image)
            if random.random() < self.prob_crop:
                transformed = self.Random_Crop(image=image, mask=label)
                image = transformed["image"]
                label = transformed["mask"]

        if self.split != "":
            image, scale, (dw_left, dw_right, dh_top, dh_bottom) = letterbox(
                image, self.resize_largest_side
            )
            label = resize_and_pad_label(
                label, scale, dw_left, dw_right, dh_top, dh_bottom
            )

            if "twin" in self.model_name:
                # Label for Drivable Area decoder
                seg_da = np.zeros_like(label, dtype=np.uint8)
                for src, dst in self.idx_remapping.twin.da.items():
                    seg_da[label == src] = dst
                seg_da = torch.from_numpy(seg_da).long()

                # Label for Lane Line decoder
                seg_ll = np.zeros_like(label, dtype=np.uint8)
                for src, dst in self.idx_remapping.twin.ll.items():
                    seg_ll[label == src] = dst
                seg_ll = torch.from_numpy(seg_ll).long()

                seg = (seg_da, seg_ll)
            elif "single" in self.model_name:
                seg = np.zeros_like(label, dtype=np.uint8)
                for src, dst in self.idx_remapping.single.items():
                    seg[label == src] = dst
                seg = torch.from_numpy(seg).long()

            image = np.array(image)
            image = image[:, :, ::-1].transpose(2, 0, 1)
            image = np.ascontiguousarray(image)

            # No need for shapes info during train/val
            shapes = []
        else:
            h_orig, w_orig = image.shape[:2]

            image, pad = letterbox_for_img(image, self.img_resize)
            shapes = (h_orig, w_orig, pad)

            image = np.array(image)
            image = image[:, :, ::-1].transpose(2, 0, 1)
            image = np.ascontiguousarray(image)

            # No need for segmentation info during inference
            seg = []

        return image_name, torch.from_numpy(image), seg, shapes
