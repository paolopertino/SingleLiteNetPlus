import logging

import torch

from tqdm import tqdm

from .SegmentationMetric import SegmentationMetric

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@torch.no_grad()
def val_one(
    val_loader=None,
    model=None,
    criteria=None,
    half=False,
    logger=None,
    epoch=0,
    verbose=True,
    device=None,
):

    model.eval()

    epoch_focal_loss = 0
    epoch_tversky_loss = 0
    epoch_tv_loss = 0
    epoch_boundary_loss = 0
    epoch_total_loss = 0

    SEG = SegmentationMetric(model.out.up_conv.deconv.out_channels)
    SEG.reset()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    if verbose:
        pbar = tqdm(pbar, total=total_batches)
    for _, (_, input_img, target, _) in pbar:
        input_img = (
            input_img.cuda().half() / 255.0
            if half
            else input_img.cuda().float() / 255.0
        )

        # run the model
        with torch.no_grad():
            if half:
                with torch.amp.autocast(device_type=device):
                    output = model(input_img)
                    focal_loss, tversky_loss, tv_loss, boundary_loss, loss = criteria(
                        output, target
                    )
            else:
                output = model(input_img)
                focal_loss, tversky_loss, tv_loss, boundary_loss, loss = criteria(
                    output, target
                )
        epoch_focal_loss += focal_loss
        epoch_tversky_loss += tversky_loss
        epoch_tv_loss += tv_loss
        epoch_boundary_loss += boundary_loss
        epoch_total_loss += loss

        ###--------------------------Segmentation-------------------------
        _, predict = torch.max(output, 1)
        gt = target

        SEG.addBatch(predict.cpu(), gt.cpu())
        ###--------------------------Segmentation-------------------------

    epoch_focal_loss /= total_batches
    epoch_tversky_loss /= total_batches
    epoch_boundary_loss /= total_batches
    epoch_tv_loss /= total_batches
    epoch_total_loss /= total_batches
    LOGGER.info(
        f"Validation - Tversky Loss: {epoch_tversky_loss:.4f}, "
        f"Focal Loss: {epoch_focal_loss:.4f}, "
        f"Total Variation Loss: {epoch_tv_loss:.4f}, "
        f"Boundary Loss: {epoch_boundary_loss:.4f}, "
        f"Total Loss: {epoch_total_loss:.4f}"
    )

    if logger is not None:
        logger.log(
            {
                "epoch": epoch,
                "val/tversky_loss": epoch_tversky_loss,
                "val/focal_loss": epoch_focal_loss,
                "val/tv_loss": epoch_tv_loss,
                "val/boundary_loss": epoch_boundary_loss,
                "val/total_loss": epoch_total_loss,
            }
        )

    return SEG
