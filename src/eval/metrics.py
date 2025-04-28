import json
import logging
import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

logger = logging.getLogger("PULSE_logger")

# TODO: maybe implement sub-group AUROC and AUPRC (considering the fairness and biases of models) for different genders, age ranges, etc.

class MetricsTracker:
    """
    A class to track and report metrics during model validation.
    """

    def __init__(
        self, model_id, task_id, dataset_name, save_dir="output", metrics_to_track=None
    ) -> None:
        """
        Initialize the metrics tracker. All tasks and datasets will be saved to the same model-metrics file.

        Args:
            model_id: Identifier for the model
            task_id: Identifier for the task
            dataset_name: Name of the dataset
            save_dir: Directory where reports will be saved
            metrics_to_track: List of metrics to track (default is a predefined list)
        """
        self.model_id = model_id
        self.task_id = task_id
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.summary = {}
        self.metrics_to_track = metrics_to_track or [
            "auroc",
            "auprc",
            "normalized_auprc",
            "sensitivity",
            "specificity",
            "f1_score",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "mcc",
        ]
        self.metrics = {metric: [] for metric in self.metrics_to_track}
        self.results = {
            "predictions": [],
            "labels": [],
        }

    def add_results(self, predictions: List, labels: List) -> None:
        """
        Add results to the metrics tracker.

        Args:
            predictions: List of predicted values
            labels: List of true labels
        """
        # Make sure that predictions and labels have the same dimensions
        labels = np.array(labels).flatten()
        predictions = np.array(predictions).flatten()

        self.results["predictions"].extend(predictions)
        self.results["labels"].extend(labels)

    def compute_overall_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for all results in tracked metrics..

        Returns:
            Dictionary containing statistics for each metric
        """
        summary = {}

        # Check if we have stored results to calculate overall metrics
        if self.results["predictions"] and self.results["labels"]:
            predictions = np.array(self.results["predictions"])
            labels = np.array(self.results["labels"])

            # Calculate overall metrics based on all predictions and labels
            overall_metrics = calculate_all_metrics(labels, predictions)

            # Store only the metrics we're tracking
            overall_summary = {
                metric: overall_metrics[metric]
                for metric in self.metrics_to_track
                if metric in overall_metrics
            }
            summary["overall"] = overall_summary

        return summary

    def save_report(self) -> str:
        """
        Generate and save a report of the tracked metrics.

        Returns:
            Path to the saved report file
        """
        os.makedirs(self.save_dir, exist_ok=True)

        # Create the report
        report = {
            "model_id": self.model_id,
            "task_id": self.task_id,
            "dataset": self.dataset_name,
            "metrics_summary": self.summary,
        }

        # Save in append mode with proper JSON formatting
        report_path = os.path.join(
            self.save_dir, f"{self.model_id}_metrics_report.json"
        )

        # Read existing data or create empty list if file doesn't exist
        existing_data = []
        if os.path.exists(report_path) and os.path.getsize(report_path) > 0:
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not decode existing JSON in {report_path}, creating new file"
                )
                existing_data = []

        # Add the new report to the list of reports
        existing_data.append(report)

        # Write the updated data back to the file
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)

        logger.info("Metrics report saved to %s", report_path)
        return report_path


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
) -> Union[float, Dict[str, float]]:
    """
    Calculate Area Under the Precision-Recall Curve (AUPRC) and Normalized AUPRC
    
    Normalized AUPRC is calculated by dividing the AUPRC by the fraction of 
    positive class samples in the total samples. This helps adjust for class imbalance.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities or scores

    Returns:
        Either a float (AUPRC) or a dictionary containing both AUPRC and normalized AUPRC
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
    auprc = auc(recall, precision)
    
    # Calculate normalized AUPRC
    positive_fraction = np.mean(y_true)
    
    if positive_fraction == 0:
        logger.warning("No positive samples in y_true. Cannot normalize AUPRC.")
        normalized_auprc = np.nan
    else:
        normalized_auprc = auprc / positive_fraction
    
    return {"auprc": auprc, "normalized_auprc": normalized_auprc}

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

def calculate_balanced_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> float:
    """
    Calculate Balanced Accuracy using sklearn's balanced_accuracy_score

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        Balanced accuracy score
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    return balanced_accuracy_score(y_true, y_pred_binary)

def calculate_mcc(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC)

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        MCC score
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    return matthews_corrcoef(y_true, y_pred_binary)

def calculate_all_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate all metrics at once

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities (0..1)
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        Dictionary containing all metrics rounded to 3 decimal places
    """
    # Get both AUPRC and normalized AUPRC in one call
    auprc_results = calculate_auprc(y_true, y_pred)
    
    metrics = {
        "auroc": calculate_auroc(y_true, y_pred),
        "auprc": auprc_results["auprc"],
        "normalized_auprc": auprc_results["normalized_auprc"],
        "sensitivity": calculate_sensitivity(y_true, y_pred, threshold),
        "specificity": calculate_specificity(y_true, y_pred, threshold),
        "f1_score": calculate_f1_score(y_true, y_pred, threshold),
        "accuracy": calculate_accuracy(y_true, y_pred, threshold),
        "balanced_accuracy": calculate_balanced_accuracy(y_true, y_pred, threshold),
        "precision": calculate_precision(y_true, y_pred, threshold),
        "recall": calculate_recall(y_true, y_pred, threshold),
        "mcc": calculate_mcc(y_true, y_pred, threshold),
    }
    
    # Round all metrics to 3 decimal places
    rounded_metrics = {k: round(v, 3) if not np.isnan(v) else v for k, v in metrics.items()}
    
    return rounded_metrics


def calc_metric_stats(metrics_tracker: dict, model_id: str, save_dir=str) -> None:
    """
    Calculate and save statistics for the tracked metrics.

    Args:
        metrics_tracker: Dictionary containing the tracked metrics
        model_id: Identifier for the model
        save_dir: Directory where the statistics will be saved
    """
    # Calculate mean and standard deviation for each metric
    stats = {}
    for metric_name, values in metrics_tracker.items():
        # Convert numpy types to native Python types for JSON serialization
        values_array = np.array(values)
        stats[metric_name] = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array)),
            "count": int(len(values_array)),
        }

    stats["model_id"] = model_id

    # Save the statistics to a file
    stats_file_path = os.path.join(save_dir, "metrics_stats.json")

    with open(stats_file_path, "w", encoding="utf-8") as f:

        json.dump(stats, f, indent=4)

    logger.info(f"Metrics statistics saved to {stats_file_path}")
