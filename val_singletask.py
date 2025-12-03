import logging

import hydra
import torch
import torch.backends.cudnn as cudnn

from omegaconf import OmegaConf

from datasets import Dataset
from evaluation import val_one
from model import netParams, SingleLiteNetPlus
from training.losses import SingleLoss
from utils import fix_randseed, log_metrics_single

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def validation(config):
    """
    Perform model validation on the BDD100K dataset.
    :param args: Parsed command-line arguments.
    """
    LOGGER.info(f"##### CONFIGURATION #####\n{OmegaConf.to_yaml(config)}\n########################")
    fix_randseed(config.seed)

    # Initialize model
    # Model initialization
    if config.model.version == 'default':
        model = SingleLiteNetPlus(
            encoder_hp=config.model.encoder,
            caam_hp=config.model.caam,
            decoder_hp=config.model.decoder,
        )
    else:
        raise ValueError(f"Unknown model version: {config.model.version}")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = model.cuda()
        cudnn.benchmark = True
    
    # Create validation data loader
    valLoader = torch.utils.data.DataLoader(
        Dataset(
            images_folder=config.dataset.images_folder,
            annotations_folder=config.dataset.annotations_folder,
            hyp=config.augmentation,
            model_name=config.model.name,
            split="val",
            lidar_img1=config.dataset.get('lidar_img1', None),
            lidar_img2=config.dataset.get('lidar_img2', None),
            lidar_img3=config.dataset.get('lidar_img3', None),
            idx_remapping=config.dataset.idx_remapping
        ),
        batch_size=config.dataset.batch_size, shuffle=False, num_workers=config.dataset.num_workers, pin_memory=True)

    # Print model parameter count
    LOGGER.info(f'Total network parameters: {netParams(model)}')
    
    # Load pretrained weights
    model.load_state_dict(torch.load(config.routine.model_weights))
    model.eval()
    LOGGER.info(f"Loaded model weights from {config.routine.model_weights}")

    criteria = SingleLoss(config.loss)
    
    # Perform validation
    metrics_computer = val_one(
        val_loader=valLoader,
        model=model,
        criteria=criteria,
        half=config.routine.mixed_precision_inference,
        verbose=config.logging.verbose,
        device=config.device
    )
    
    _ = log_metrics_single(
        logger=None, # We want to print only the metrics to screen
        metrics=metrics_computer,
        epoch=0,
        class2idx=config.dataset.class2idx,
        idx_remapping=config.dataset.idx_remapping,
        split='val'
    )

if __name__ == '__main__':
    validation()
