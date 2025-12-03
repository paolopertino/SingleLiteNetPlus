import logging
import logging.config
import os

import torch

from tqdm import tqdm

LOGGING_NAME = "custom"


def set_logging(name=LOGGING_NAME, verbose=True):
    rank = int(os.getenv("RANK", -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {
                name: {
                    "level": level,
                    "handlers": [name],
                    "propagate": False,
                }
            },
        }
    )


set_logging(LOGGING_NAME)
LOGGER = logging.getLogger(LOGGING_NAME)


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scaler,
    device,
    mixed_precision_training,
    verbose=False,
    logger=None,
):
    model.train()
    LOGGER.info("epoch: %d", epoch)
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)

    # Logging the losses per epoch
    epoch_focal_loss = 0
    epoch_tversky_loss = 0
    epoch_tv_loss = 0
    epoch_boundary_loss = 0
    epoch_total_loss = 0

    if verbose:
        LOGGER.info(
            ("\n" + "%13s" * 6)
            % (
                "Epoch",
                "TverskyLoss",
                "FocalLoss",
                "TotalVariationLoss",
                "BoundaryLoss",
                "TotalLoss",
            )
        )
        pbar = tqdm(pbar, total=total_batches, bar_format="{l_bar}{bar:10}{r_bar}")
    for i, (_, input_imgs, target, _) in pbar:
        optimizer.zero_grad()
        if "cuda" in device:
            input_imgs = input_imgs.to(device).float()

        input_imgs = input_imgs / 255.0  # Normalize input images to [0, 1]

        if mixed_precision_training:
            with torch.amp.autocast(device_type=device):
                output = model(input_imgs)
                focal_loss, tversky_loss, tv_loss, boundary_loss, loss = criterion(
                    output, target
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(input_imgs)
            focal_loss, tversky_loss, tv_loss, boundary_loss, loss = criterion(
                output, target
            )
            loss.backward()
            optimizer.step()

        # Update epoch losses
        epoch_focal_loss += focal_loss
        epoch_tversky_loss += tversky_loss
        epoch_tv_loss += tv_loss
        epoch_boundary_loss += boundary_loss
        epoch_total_loss += loss

        if verbose:
            pbar.set_description(
                ("%13s" * 1 + "%13.4g" * 5)
                % (
                    f"{epoch}/{300 - 1}",
                    tversky_loss,
                    focal_loss,
                    tv_loss,
                    boundary_loss,
                    loss.item(),
                )
            )

    # Log epoch losses
    epoch_focal_loss /= total_batches
    epoch_tversky_loss /= total_batches
    epoch_tv_loss /= total_batches
    epoch_boundary_loss /= total_batches
    epoch_total_loss /= total_batches
    LOGGER.info(
        f"Epoch {epoch} - Tversky Loss: {epoch_tversky_loss:.4f}, "
        f"Focal Loss: {epoch_focal_loss:.4f}, "
        f"Total Variation Loss: {epoch_tv_loss:.4f}, "
        f"Boundary Loss: {epoch_boundary_loss:.4f}, "
        f"Total Loss: {epoch_total_loss:.4f}"
    )

    if logger is not None:
        logger.log(
            {
                "epoch": epoch,
                "train/tversky_loss": epoch_tversky_loss,
                "train/focal_loss": epoch_focal_loss,
                "train/tv_loss": epoch_tv_loss,
                "train/boundary_loss": epoch_boundary_loss,
                "train/total_loss": epoch_total_loss,
            }
        )

    return None
