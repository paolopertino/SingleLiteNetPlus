import logging

import os

import numpy as np

from evaluation import SegmentationMetric

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def args_to_dict(args):
    """Convert argparse.Namespace to dictionary, handling nested objects."""
    if hasattr(args, "__dict__"):
        return vars(args)
    else:
        return args


def prepare_wandb_config(args, hyp):
    """Prepare configuration dictionary for wandb logging."""
    # Convert args to dictionary
    args_dict = args_to_dict(args)

    # Create base config
    dataset_name = os.path.basename(os.path.normpath(args.dataset_root_path))
    config = {
        "description": f"Finetuning of TwinLiteNetPlus on {dataset_name}",
        "architecture": "TwinLiteNetPlus",
        "dataset_name": dataset_name,
    }

    # Add hyperparameters with 'hyp_' prefix to avoid conflicts
    for key, value in hyp.items():
        config[f"hyp_{key}"] = value

    # Add args with 'args_' prefix to avoid conflicts
    for key, value in args_dict.items():
        # Skip non-serializable objects or sensitive info
        if key in ["device"]:  # Skip device object
            config[f"args_{key}"] = str(value)
        elif isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            config[f"args_{key}"] = value
        else:
            # Convert other objects to string representation
            try:
                config[f"args_{key}"] = str(value)
            except Exception as e:
                config[f"args_{key}"] = f"Error: {str(e)}"

    return config


def log_metrics(
    logger,
    metrics_da: SegmentationMetric,
    metrics_ll: SegmentationMetric,
    epoch: int,
    class2idx,
    idx_remapping,
    split: str = "val",
) -> dict[str, float]:
    metrics = {
        "epoch": epoch,
    }
    ll_indices = list({v for _, v in idx_remapping.twin.ll.items()})
    da_indices = list({v for _, v in idx_remapping.twin.da.items()})

    iou_da = metrics_da.IntersectionOverUnion()
    acc_da = metrics_da.pixelAccuracy()

    iou_ll = metrics_ll.IntersectionOverUnion()
    acc_ll = metrics_ll.lineAccuracy(pos_classes=tuple(ll_indices))

    # Saving the metrics for each class
    for cls_name, cls_idx in class2idx.items():
        if cls_idx == 0:
            continue  # Skip background class

        if cls_name.startswith("da"):
            metrics[f"{split}/{cls_name}_iou"] = iou_da[idx_remapping.twin.da[cls_idx]]
        elif cls_name.startswith("ll"):
            metrics[f"{split}/{cls_name}_iou"] = iou_ll[idx_remapping.twin.ll[cls_idx]]
        else:
            raise ValueError(f"Unknown class name: {cls_name}")

    # Saving the accuracies
    metrics[f"{split}/da_acc"] = acc_da
    metrics[f"{split}/ll_acc"] = acc_ll

    # Saving aggregated metrics for da and ll both with and without background
    iou_da_no_bg = np.nanmean(iou_da[da_indices])
    iou_da_with_bg = np.nanmean(iou_da)
    iou_ll_no_bg = np.nanmean(iou_ll[ll_indices])
    iou_ll_with_bg = np.nanmean(iou_ll)
    metrics[f"{split}/da_miou"] = iou_da_no_bg
    metrics[f"{split}/ll_miou"] = iou_ll_no_bg
    metrics[f"{split}/da_miou_bg"] = iou_da_with_bg
    metrics[f"{split}/ll_miou_bg"] = iou_ll_with_bg

    # Logging
    LOGGER.info("##### EVALUATION #####")
    for key, value in metrics.items():
        LOGGER.info(f"{key}: {value:.4f}")

    if logger is not None:
        logger.log(metrics)

    return metrics


def log_metrics_single(
    logger,
    metrics: SegmentationMetric,
    epoch: int,
    class2idx,
    idx_remapping,
    split: str = "val",
) -> dict[str, float]:
    metrics_out = {
        "epoch": epoch,
    }
    da_classes = {v for k, v in class2idx.items() if k.startswith("da")}
    da_indices = list({idx_remapping.single[v] for v in da_classes})
    ll_classes = {v for k, v in class2idx.items() if k.startswith("ll")}
    ll_indices = list({idx_remapping.single[v] for v in ll_classes})

    iou = metrics.IntersectionOverUnion()
    acc = metrics.pixelAccuracy()
    ll_acc = metrics.lineAccuracy(pos_classes=tuple(ll_indices))

    # Saving the metrics for each class
    for cls_name, cls_idx in class2idx.items():
        if cls_idx == 0:
            continue  # Skip background class

        remapped_idx = idx_remapping.single[cls_idx]
        metrics_out[f"{split}/{cls_name}_iou"] = iou[remapped_idx]

    # Saving the accuracies
    metrics_out[f"{split}/acc"] = acc
    metrics_out[f"{split}/ll_acc"] = ll_acc

    # Saving aggregated metrics for da and ll both with and without background
    iou_da_no_bg = np.nanmean(iou[da_indices])
    iou_da_with_bg = np.nanmean(iou[[0] + da_indices])
    iou_ll_no_bg = np.nanmean(iou[ll_indices])
    iou_ll_with_bg = np.nanmean(iou[[0] + ll_indices])
    global_miou = np.nanmean(iou[1:])
    global_miou_with_bg = np.nanmean(iou)
    metrics_out[f"{split}/da_miou"] = iou_da_no_bg
    metrics_out[f"{split}/ll_miou"] = iou_ll_no_bg
    metrics_out[f"{split}/da_miou_bg"] = iou_da_with_bg
    metrics_out[f"{split}/ll_miou_bg"] = iou_ll_with_bg
    metrics_out[f"{split}/miou"] = global_miou
    metrics_out[f"{split}/miou_bg"] = global_miou_with_bg

    # Logging
    LOGGER.info("##### EVALUATION #####")
    for key, value in metrics_out.items():
        LOGGER.info(f"{key}: {value:.4f}")

    if logger is not None:
        logger.log(metrics_out)

    return metrics_out
