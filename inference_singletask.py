import logging
import os
import time

from pathlib import Path

import cv2
import hydra
import numpy as np
import torch

from omegaconf import OmegaConf
from tqdm import tqdm

from datasets import Dataset
from model import SingleLiteNetPlus

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def show_seg_result(img, result):
    color_area = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)

    color_area[result == 1] = [0, 255, 0]
    color_area[result == 2] = [0, 0, 255]
    color_area[result == 3] = [255, 0, 0]
    color_area[result == 4] = [255, 128, 0]
    color_area[result == 5] = [0, 255, 128]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img = cv2.resize(
        img, (color_seg.shape[1], color_seg.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)

    return img


@torch.no_grad()
def segment_image(image, shapes, model, device, half, idx_remapping):
    if image.ndimension() == 3:
        image = image.unsqueeze(0)

    # Move to CUDA and normalize
    image = (
        image.to(device).half() / 255.0 if half else image.to(device).float() / 255.0
    )

    # Retrieving padding information
    original_height, original_width, pad = shapes[0].item(), shapes[1].item(), shapes[2]
    _, _, image_h, image_w = image.shape

    pad_w, pad_h = pad[0].item(), pad[1].item()
    pad_top = int(round(pad_h - 0.1))
    pad_bottom = int(round(pad_h + 0.1))
    pad_left = int(round(pad_w - 0.1))
    pad_right = int(round(pad_w + 0.1))

    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = img_np[
        pad_top : (image_h - pad_bottom), pad_left : (image_w - pad_right), :
    ]
    img_np = (img_np * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    start_time = time.time()

    if half:
        with torch.amp.autocast(device_type=device):
            seg_out = model(image)
    else:
        seg_out = model(image)

    predict = seg_out[
        :, :, pad_top : (image_h - pad_bottom), pad_left : (image_w - pad_right)
    ]

    seg_mask = torch.nn.functional.interpolate(
        predict,
        size=(original_height, original_width),
        mode="bilinear",
    )
    _, seg_mask = torch.max(seg_mask, 1)
    seg_mask = seg_mask.int().squeeze().cpu().numpy()

    # Reverse mapping for inference
    def reverse_mapping(mapping: dict):
        reversed_map = {}
        for src, dst in mapping.items():
            # Only add if not already present to keep the first match
            if dst not in reversed_map:
                reversed_map[dst] = src
        return reversed_map

    final_mask = np.zeros_like(seg_mask, dtype=np.uint8)

    # Reverse the mapping
    rev = reverse_mapping(idx_remapping.single)

    # Apply reversed mapping
    for src, dst in rev.items():
        final_mask[seg_mask == src] = dst

    end_time = time.time()
    LOGGER.info(f"FPS: {1/(end_time - start_time):.2f} FPS")

    img_vis = show_seg_result(img_np, final_mask)

    return final_mask, img_vis


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def segment(config):
    LOGGER.info(
        f"##### CONFIGURATION #####\n{OmegaConf.to_yaml(config)}\n########################"
    )

    device = config.device
    half = config.routine.mixed_precision_inference

    # Model initialization
    if config.model.version == "default":
        model = SingleLiteNetPlus(
            encoder_hp=config.model.encoder,
            caam_hp=config.model.caam,
            decoder_hp=config.model.decoder,
        )
    else:
        raise ValueError(f"Unknown model version: {config.model.version}")

    model = model.to(device)
    model.load_state_dict(torch.load(config.routine.model_weights))
    model.eval()

    if half:
        model.half()  # to FP16

    dataset = torch.utils.data.DataLoader(
        Dataset(
            images_folder=config.dataset.images_folder,
            annotations_folder=None,
            hyp=None,
            model_name=config.model.name,
            split="",
            lidar_img1=config.dataset.get("lidar_img1", None),
            lidar_img2=config.dataset.get("lidar_img2", None),
            lidar_img3=config.dataset.get("lidar_img3", None),
            idx_remapping=config.dataset.idx_remapping,
            img_resize=config.dataset.img_resize,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
    )

    # Logging setup
    os.makedirs(config.savedir, exist_ok=True)
    os.makedirs(os.path.join(config.savedir, "predictions"), exist_ok=True)

    pbar = tqdm(enumerate(dataset), total=len(dataset))
    for _, (image_name, image, _, shapes) in pbar:
        image_name = image_name[0]

        # Image segmentation
        final_mask, img_vis = segment_image(
            image, shapes, model, device, half, config.dataset.idx_remapping
        )

        # Saving the results
        save_path = str(config.savedir + "/" + Path(image_name).name)
        mask_save_path = str(
            config.savedir + "/predictions/" + Path(image_name).stem + ".png"
        )

        cv2.imwrite(mask_save_path, final_mask.astype(np.uint8))
        cv2.imwrite(save_path, img_vis)


if __name__ == "__main__":
    with torch.no_grad():
        segment()
