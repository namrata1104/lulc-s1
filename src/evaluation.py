"""Confusion-matrix based metrics: OA, mIoU, per-class IoU / precision / recall / F1."""
import numpy as np
import pandas as pd

from . import config as C

NUM_CLASSES = len(C.CLASSES)


def confusion(y_true, y_pred, n=NUM_CLASSES):
    """Rows = reference, columns = prediction. IGNORE pixels are skipped."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    keep = y_true != C.IGNORE
    return np.bincount(n * y_true[keep].astype(np.int64) + y_pred[keep], minlength=n * n).reshape(n, n)


def scores(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    ref, pred = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (ref + pred - tp)
        precision = tp / pred
        recall = tp / ref
        f1 = 2 * tp / (ref + pred)
    per_class = pd.DataFrame({"IoU": iou, "precision": precision, "recall": recall, "F1": f1,
                              "support_%": 100 * ref / ref.sum()}, index=C.CLASSES)
    present = ref > 0
    return dict(OA=tp.sum() / cm.sum(), mIoU=np.nanmean(iou[present]),
                mF1=np.nanmean(f1[present]), per_class=per_class)
