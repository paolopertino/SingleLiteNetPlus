import numpy as np
import torch


class SegmentationMetric(object):
    def __init__(self, numClass: int):
        self.numClass = int(numClass)
        self.reset()

    def reset(self):
        # Keep counts as int64; cast to float only when computing ratios
        self.confusionMatrix = np.zeros((self.numClass, self.numClass), dtype=np.int64)

    @staticmethod
    def _to_numpy_int(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        # Ensure integer type for bincount indexing
        if not np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.int64, copy=False)
        return x

    def genConfusionMatrix(self, imgPredict, imgLabel):
        pred = self._to_numpy_int(imgPredict)
        lab = self._to_numpy_int(imgLabel)

        assert pred.shape == lab.shape, "Prediction/label shapes must match"

        # Keep only valid labels; ignore negatives and labels >= numClass
        mask = (lab >= 0) & (lab < self.numClass)

        # Flatten via boolean mask
        lab_m = lab[mask].ravel()
        pred_m = pred[mask].ravel()

        # Map (lab, pred) pairs to [0, numClass^2-1]
        label = self.numClass * lab_m + pred_m
        count = np.bincount(label, minlength=self.numClass**2)
        return count.reshape(self.numClass, self.numClass)

    def addBatch(self, imgPredict, imgLabel):
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def pixelAccuracy(self):
        total = self.confusionMatrix.sum()
        if total == 0:
            return 0.0
        correct = np.trace(self.confusionMatrix)
        return float(correct) / float(total)

    def IntersectionOverUnion(self):
        cm = self.confusionMatrix.astype(np.float64, copy=False)
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = intersection / union
        # Keep NaNs; let callers decide whether to ignore them
        return iou  # shape: (numClass,)

    def lineAccuracy(self, pos_classes=(1,)):
        """
        Balanced accuracy for 'line vs not-line'.
        pos_classes: iterable of class indices considered positive (e.g., (1,) or (1,2))
        """
        pos = np.zeros(self.numClass, dtype=bool)
        pos[list(pos_classes)] = True
        neg = ~pos

        cm = self.confusionMatrix.astype(np.float64, copy=False)

        tp = cm[np.ix_(pos, pos)].sum()
        fn = cm[np.ix_(pos, neg)].sum()
        tn = cm[np.ix_(neg, neg)].sum()
        fp = cm[np.ix_(neg, pos)].sum()

        sens = tp / (tp + fn + 1e-12)  # TPR
        spec = tn / (tn + fp + 1e-12)  # TNR
        return float((sens + spec) / 2.0)
