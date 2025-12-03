import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int = 7, verbose: bool = False, mode: str = "min"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_epoch = 0
        self.best_checkpoint = None
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, score: float, epoch: int, checkpoint=None):
        if self.mode == "min":
            if self.best_score is None:
                self.best_epoch = epoch
                self.best_checkpoint = checkpoint
                self.best_score = score
            elif score > self.best_score:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_epoch = epoch
                self.best_checkpoint = checkpoint
                self.best_score = score
                self.counter = 0
        elif self.mode == "max":
            if self.best_score is None:
                self.best_epoch = epoch
                self.best_checkpoint = checkpoint
                self.best_score = score
            elif score < self.best_score:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_epoch = epoch
                self.best_checkpoint = checkpoint
                self.best_score = score
                self.counter = 0

        if self.verbose:
            if self.early_stop:
                LOGGER.info(
                    f"Early stopping triggered after {self.counter} epochs without improvement."
                )
            else:
                LOGGER.info(
                    f"Current best score: {self.best_score}, patience counter: {self.counter}/{self.patience}"
                )

        return self.early_stop


class MultiMetricsEarlyStopping:
    """Early stops the training if multiple metrics don't improve after a given patience."""

    def __init__(self, patience: int = 7, verbose: bool = False, modes: dict = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_epoch = 0
        self.best_checkpoint = None
        self.best_scores = {}
        self.early_stop = False
        self.modes = modes if modes is not None else {}

    def _is_improvement(self, metric_name, current_score):
        mode = self.modes.get(metric_name, "max")
        best_score = self.best_scores.get(metric_name)

        if best_score is None:
            return True  # Always accept first score

        if mode == "min":
            return current_score < best_score
        else:  # default to 'max'
            return current_score > best_score

    def __call__(self, metrics: dict, epoch: int, checkpoint):
        improvements = []
        for metric_name, current_score in metrics.items():
            if self._is_improvement(metric_name, current_score):
                if self.verbose:
                    LOGGER.info(
                        f"[EarlyStopping] {metric_name} improved "
                        f"from {self.best_scores.get(metric_name, 'N/A')} to {current_score}"
                    )
                self.best_scores[metric_name] = current_score
                improvements.append(True)
            else:
                improvements.append(False)

        if any(improvements):
            self.counter = 0
            self.best_epoch = epoch
            self.best_checkpoint = checkpoint
        else:
            self.counter += 1
            if self.verbose:
                LOGGER.info(
                    f"[EarlyStopping] No improvement in any metric. "
                    f"Counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                if self.verbose:
                    LOGGER.info("[EarlyStopping] Early stopping triggered.")
                self.early_stop = True
        return self.early_stop
