import logging
import os

from datetime import datetime

import hydra
import torch

try:
    import wandb
except:
    wandb = None

from omegaconf import OmegaConf

from datasets import Dataset
from evaluation import val_one
from model import SingleLiteNetPlus
from training import train
from training.callbacks import (
    save_checkpoint,
    MultiMetricsEarlyStopping,
    FineTuningScheduler,
)
from training.losses import SingleLoss
from training.lr_schedulers import poly_lr_scheduler
from utils import fix_randseed, log_metrics_single

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_data(config):
    train_loader = torch.utils.data.DataLoader(
        Dataset(
            images_folder=config.dataset.images_folder,
            annotations_folder=config.dataset.annotations_folder,
            hyp=config.augmentation,
            model_name=config.model.name,
            split="train",
            lidar_img1=config.dataset.get("lidar_img1", None),
            lidar_img2=config.dataset.get("lidar_img2", None),
            lidar_img3=config.dataset.get("lidar_img3", None),
            idx_remapping=config.dataset.idx_remapping,
        ),
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        Dataset(
            images_folder=config.dataset.images_folder,
            annotations_folder=config.dataset.annotations_folder,
            hyp=config.augmentation,
            model_name=config.model.name,
            split="val",
            lidar_img1=config.dataset.get("lidar_img1", None),
            lidar_img2=config.dataset.get("lidar_img2", None),
            lidar_img3=config.dataset.get("lidar_img3", None),
            idx_remapping=config.dataset.idx_remapping,
        ),
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


@hydra.main(version_base=None, config_path="configs", config_name="config")
def finetune_net(config):
    LOGGER.info(
        f"##### CONFIGURATION #####\n{OmegaConf.to_yaml(config)}\n########################"
    )
    dict_configs = OmegaConf.to_container(config, resolve=True)
    # Setup savedir
    os.makedirs(config.savedir, exist_ok=True)

    # Setup wandb if enabled
    if config.logging.wandb and wandb:
        # The dataset name is the second to last part of the images folder path. e.g., if the path is '/path/to/dataset/AIDA/images', the dataset name will be 'AIDA'.
        dataset_name = os.path.basename(
            os.path.dirname(os.path.normpath(config.dataset.images_folder))
        )
        wandb.login()
        wandb.init(
            project="TwinLiteNetPlus",
            name=f"{config.logging.exp_name}_{dataset_name}_{config.model.name}_{str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))}",
            config=dict_configs,
        )

    fix_randseed(config.seed)

    train_loader, val_loader = load_data(config)

    # Initializing model, optimizer, and loss function
    # Model initialization
    if config.model.version == "default":
        model = SingleLiteNetPlus(
            encoder_hp=config.model.encoder,
            caam_hp=config.model.caam,
            decoder_hp=config.model.decoder,
        )
    else:
        raise ValueError(f"Unknown model version: {config.model.version}")

    model.load_state_dict(torch.load(config.routine.model_weights))
    for param in model.parameters():
        param.requires_grad = False

    ft_schedule = {
        **config.routine.ft_schedule,
    }

    # Finetuning Scheduler setup
    ft_scheduler = FineTuningScheduler(schedule=ft_schedule)

    model = model.to(config.device)
    criteria = SingleLoss(config.loss)
    lr = config.routine.lr
    scaler = (
        torch.amp.GradScaler(device=config.device)
        if config.routine.mixed_precision_training
        else None
    )

    if config.logging.wandb and wandb:
        # The log_freq should be at each epoch, so it should be equal to the number of batches
        # divided by the batch size.
        total_batches = len(train_loader)
        log_freq = total_batches

        wandb.watch(model, log="all", log_freq=log_freq)
        LOGGER.info("WandB logging enabled, model parameters will be logged.")

    # Early stopping setup
    early_stopping = MultiMetricsEarlyStopping(
        patience=config.routine.early_stopping_patience,
        verbose=config.logging.verbose,
        modes={"val_da_miou": "max", "val_ll_iou": "max"},
    )

    # Train loop
    start_epoch = 0
    for epoch in range(start_epoch, config.routine.max_epochs):
        if ft_scheduler.step(epoch, model):
            # Reinitialize optimizer with the new parameters
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                betas=(config.optimizer.momentum, 0.999),
                eps=config.optimizer.eps,
                weight_decay=config.optimizer.weight_decay,
            )

        model_file_name = os.path.join(config.savedir, f"model_{epoch}.pth")
        poly_lr_scheduler(
            config.routine.max_epochs, config.routine.lr, optimizer, epoch
        )
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        LOGGER.info(f"Learning rate: {lr}")

        model.train()
        train(
            train_loader=train_loader,
            model=model,
            criterion=criteria,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
            device=config.device,
            mixed_precision_training=config.routine.mixed_precision_training,
            verbose=config.logging.verbose,
            logger=wandb if config.logging.wandb and wandb else None,
        )

        model.eval()
        metrics_computer = val_one(
            val_loader=val_loader,
            model=model,
            criteria=criteria,
            verbose=config.logging.verbose,
            half=config.routine.mixed_precision_training,
            logger=wandb if config.logging.wandb and wandb else None,
            epoch=epoch,
            device=config.device,
        )

        metrics = log_metrics_single(
            logger=wandb if config.logging.wandb and wandb else None,
            metrics=metrics_computer,
            epoch=epoch,
            class2idx=config.dataset.class2idx,
            idx_remapping=config.dataset.idx_remapping,
            split="val",
        )

        torch.save(model.state_dict(), model_file_name)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr": lr,
            },
            os.path.join(config.savedir, "checkpoint.pth.tar"),
        )

        # Check for early stopping
        if early_stopping(
            {
                "val_da_miou": metrics["val/da_miou"],
                "val_ll_iou": metrics["val/ll_miou"],
            },
            epoch=epoch,
            checkpoint=model.state_dict(),
        ):
            LOGGER.info(
                f"Early stopping triggered. Best epoch: {early_stopping.best_epoch}"
            )

            # Saving the best model
            best_model_path = os.path.join(config.savedir, "best_model.pth")
            torch.save(early_stopping.best_checkpoint, best_model_path)
            LOGGER.info(f"Best model saved at {best_model_path}")
            break

    if config.logging.wandb and wandb:
        wandb.finish()


if __name__ == "__main__":
    finetune_net()
