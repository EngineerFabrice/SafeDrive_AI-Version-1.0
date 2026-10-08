"""Classification metrics with "alcoholic" as the positive (potentially not sober) class."""
import numpy as np


def binary_metrics(y_true, prob_pos, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float)
    y_pred = (prob_pos >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0          # sensitivity for the positive class
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {"n": n, "accuracy": (tp + tn) / n if n else 0.0, "precision": precision, "recall": recall,
           "sensitivity_positive": recall, "specificity_negative": specificity, "f1": f1,
           "false_positive_rate": 1 - specificity if tn + fp else 0.0,
           "false_negative_rate": 1 - recall if tp + fn else 0.0,
           # rows = true [non_alcoholic, alcoholic], cols = predicted
           "confusion_matrix": [[tn, fp], [fn, tp]], "threshold": threshold}
    if len(set(y_true.tolist())) == 2:
        from sklearn.metrics import roc_auc_score
        out["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
