import logging

import torch

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FineTuningScheduler:
    def __init__(self, schedule: dict):
        self.schedule = schedule
        self.unfrozen = set()

    def step(self, epoch: int, model: torch.nn.Module) -> bool:
        has_changed = False
        for name, unfreeze_epoch in self.schedule.items():
            if name in self.unfrozen:
                continue
            if unfreeze_epoch >= 0 and epoch == unfreeze_epoch:
                module = self._find_module(model, name)

                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True

                    LOGGER.info(f"[FineTuningScheduler] Unfroze {name} at epoch {epoch}")
                    self.unfrozen.add(name)
                    has_changed = True
                else:
                    LOGGER.warning(
                        f"[FineTuningScheduler] WARNING: Module '{name}' not found in model."
                    )

        return has_changed

    def _find_module(self, model: torch.nn.Module, name: str) -> torch.nn.Module:
        for module_name, module in model.named_modules():
            if module_name == name:
                return module
        return None
