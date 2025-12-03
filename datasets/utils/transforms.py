import random

import cv2
import numpy as np
import math

from PIL import Image
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=True, stride=32):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    # Randomizing the position of the padding
    # Makes the network invariant to the position of the padding
    random_pad_ratio_H = random.uniform(0.0, 1.0)
    random_pad_ratio_W = random.uniform(0.0, 1.0)
    dh_top = int(round(dh * random_pad_ratio_H - 0.1))
    dh_bottom = int(round(dh * (1 - random_pad_ratio_H) + 0.1))
    dw_left = int(round(dw * random_pad_ratio_W - 0.1))
    dw_right = int(round(dw * (1 - random_pad_ratio_W) + 0.1))

    image = cv2.copyMakeBorder(
        image, dh_top, dh_bottom, dw_left, dw_right, cv2.BORDER_CONSTANT, value=color
    )

    return image, r, (dw_left, dw_right, dh_top, dh_bottom)


def letterbox_for_img(
    image, new_shape=(640, 640), color=(114, 114, 114), auto=True, stride=32
):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_AREA)

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    # Equally split padding
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image, (dw, dh)


def resize_and_pad_label(label, scale, dw_left, dw_right, dh_top, dh_bottom):
    h, w = label.shape[:2]
    new_size = (int(round(w * scale)), int(round(h * scale)))

    label = cv2.resize(label, new_size, interpolation=cv2.INTER_NEAREST)
    label = cv2.copyMakeBorder(
        label, dh_top, dh_bottom, dw_left, dw_right, cv2.BORDER_CONSTANT, value=0
    )

    return label


def RandomBilateralBlur(img, sigma_bila_low=0.05, sigma_bila_high=1.0):
    """
    Apply Bilateral Filtering
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sigma = random.uniform(sigma_bila_low, sigma_bila_high)
    blurred_img = denoise_bilateral(
        np.array(img_rgb), sigma_spatial=sigma, channel_axis=2
    )
    blurred_img *= 255
    blurred_img_rgb = Image.fromarray(blurred_img.astype(np.uint8))
    blurred_img_bgr = cv2.cvtColor(np.array(blurred_img_rgb), cv2.COLOR_RGB2BGR)
    return blurred_img_bgr


def RandomGaussianBlur(img, sigma_gaus_a=1.15, sigma_gaus_b=0.15):
    """
    Apply Gaussian Blur
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sigma = sigma_gaus_b + random.random() * sigma_gaus_a
    blurred_img = gaussian(np.array(img_rgb), sigma=sigma, channel_axis=2)
    blurred_img *= 255
    blurred_img_rgb = Image.fromarray(blurred_img.astype(np.uint8))
    blurred_img_bgr = cv2.cvtColor(np.array(blurred_img_rgb), cv2.COLOR_RGB2BGR)
    return blurred_img_bgr


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """
    Change color hue, saturation and value
    """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def random_perspective(
    combination,
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    """
    Combination of image transformations
    """
    img, label = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.5 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
            label = cv2.warpPerspective(label, M, dsize=(width, height), borderValue=0)
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )
            label = cv2.warpAffine(label, M[:2], dsize=(width, height), borderValue=0)

    return (img, label)
