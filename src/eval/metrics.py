import numpy as np
import torch
from typing import Dict, Union, List, Any
import logging

from sklearn.metrics import (
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
)

logger = logging.getLogger("PULSE_logger")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Dummy function
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_auroc(
    y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate Area Under the Receiver Operating Characteristic curve (AUROC)

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities or scores

    Returns:
        AUROC score
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check if more than one class is present
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true. Returning 0.0")
        return 0.0

    auroc = roc_auc_score(y_true, y_pred)
    if np.isnan(auroc):
        logger.warning("AUROC is NaN. Returning 0.0")
        return 0.0

    return roc_auc_score(y_true, y_pred)


def calculate_auprc(
    y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate Area Under the Precision-Recall Curve (AUPRC)

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities or scores

    Returns:
        AUPRC score
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check if more than one class is present
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true. Returning NaN")
        return np.nan

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


def calculate_sensitivity(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> float:
    """
    Calculate Sensitivity (Recall or True Positive Rate)

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        Sensitivity score
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def calculate_specificity(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> float:
    """
    Calculate Specificity (True Negative Rate)

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        Specificity score
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def calculate_f1_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> float:
    """
    Calculate F1 Score

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        F1 score
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    return f1_score(y_true, y_pred_binary, labels=[0, 1], zero_division=0)


def calculate_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> float:
    """
    Calculate Accuracy

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        Accuracy score
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    return accuracy_score(y_true, y_pred_binary)


def calculate_precision(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> float:
    """
    Calculate Precision (Positive Predictive Value)

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        Precision score
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    return precision_score(y_true, y_pred_binary, zero_division=0, labels=[0, 1])


def calculate_recall(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> float:
    """
    Calculate Recall (Sensitivity)

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        Recall score
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    return recall_score(y_true, y_pred_binary, zero_division=0, labels=[0, 1])


def calculate_all_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate all metrics at once

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        Dictionary containing all metrics
    """
    return {
        "auroc": calculate_auroc(y_true, y_pred),
        "auprc": calculate_auprc(y_true, y_pred),
        "sensitivity": calculate_sensitivity(y_true, y_pred, threshold),
        "specificity": calculate_specificity(y_true, y_pred, threshold),
        "f1_score": calculate_f1_score(y_true, y_pred, threshold),
        "accuracy": calculate_accuracy(y_true, y_pred, threshold),
        "precision": calculate_precision(y_true, y_pred, threshold),
        "recall": calculate_recall(y_true, y_pred, threshold),
    }
