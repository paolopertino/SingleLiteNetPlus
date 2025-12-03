import re

import cv2
import numpy as np


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def visualize_mask_on_image(image, mask):
    """
    Visualize the mask on the image.

    Args:
        image: Original image as a numpy array.
        mask: Mask as a numpy array where each pixel value corresponds to a class ID.
    Returns:
        numpy array: Image with the mask visualized.
    """
    # Create a color mask based on the class IDs
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    # Class colors in BGR format
    class_colors = {
        1: (0, 255, 0),  # current_lane - Green
        2: (255, 0, 0),  # alternative_lane - Blue
        3: (0, 0, 255),  # line - Red
        4: (0, 165, 255),  # dashed_line - Orange
        5: (255, 165, 0),  # curb - Light Blue
    }

    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color

    # Blend the original image with the color mask
    blended_image = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

    return blended_image
