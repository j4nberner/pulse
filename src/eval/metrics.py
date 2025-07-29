import json
import logging
import os
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

logger = logging.getLogger("PULSE_logger")


class MetricsTracker:
    """
    A class to track and report metrics during model validation.
    """

    def __init__(
        self,
        model_id,
        task_id,
        dataset_name,
        save_dir="output",
        metrics_to_track=None,
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
        # Strip "Model" suffix from model_id if present
        if isinstance(model_id, str) and model_id.lower().endswith("model"):
            self.model_id = model_id[:-5]  # Remove last 5 characters ("Model")
        else:
            self.model_id = model_id

        self.task_id = task_id
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.run_id = save_dir.split("/")[-1].split("\\")[-1]
        self.summary = {}
        self.metrics_to_track = metrics_to_track or [
            "auroc",
            "auprc",
            "normalized_auprc",
            "specificity",
            "f1_score",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "mcc",
            "kappa",
            "minpse",
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

    def add_metadata_item(self, item: Dict[str, Any]) -> None:
        """
        Add a single metadata item to the tracker.

        Args:
            item: Dictionary containing keys such as 'input', 'target', 'prediction',
                  'token_time', 'infer_time', 'num_input_tokens', 'num_output_tokens', etc.
        """
        # Store additional metadata if needed
        if not hasattr(self, "items"):
            self.items = []
        self.items.append(item)

    def log_metadata(self, save_to_file: bool = True) -> None:
        """
        Print and summarize metadata for the tracked items.

        Args:
            save_to_file: If True, save the metadata summary to a file
        """
        if hasattr(self, "items") and self.items:
            df = pd.DataFrame(self.items)
            self.items = []  # Clear items after logging

            # Compute mean for numeric columns only
            means = df.select_dtypes(include=[np.number]).mean().to_dict()
            for k, v in means.items():
                logger.info("Average %s: %s", k, v)
                logger.debug("Max %s: %s", k, df[k].max())
                logger.debug("Min %s: %s", k, df[k].min())
                logger.info("Total %s: %s", k, df[k].sum())
            if save_to_file:
                summary_path = os.path.join(
                    self.save_dir,
                    f"{self.model_id}_{self.task_id}_{self.dataset_name}_{self.run_id}_metadata.csv",
                )
                logger.info(f"Saving Metadata to {summary_path}")
                # If file exists, append without header; else, write with header
                if os.path.exists(summary_path):
                    df.to_csv(summary_path, mode="a", header=False, index=False)
                else:
                    df.to_csv(summary_path, index=False)

        else:
            logger.warning("No metadata items to print.")

    def compute_overall_metrics(self) -> Dict[str, Any]:
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

    def save_report(self, **kwargs) -> str:
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
            "prompting_id": kwargs.get("prompting_id", ""),
            "run_id": self.run_id,
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
                    "Could not decode existing JSON in %s, creating new file",
                    report_path,
                )
                existing_data = []

        # Add the new report to the list of reports
        existing_data.append(report)

        # Write the updated data back to the file
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)

        # Save labels and predictions together as csv
        predictions_path = os.path.join(
            self.save_dir,
            f"{self.model_id}_{self.task_id}_{self.dataset_name}_{self.run_id}_predictions.csv",
        )
        df_predictions = pd.DataFrame(
            {
                "predictions": self.results["predictions"],
                "labels": self.results["labels"],
            }
        )
        df_predictions.to_csv(predictions_path, index=False)

        logger.info("Predictions and labels saved to %s", predictions_path)
        logger.info("Metrics report saved to %s", report_path)
        return report_path


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
) -> Dict[str, float]:
    """
    Calculate Area Under the Precision-Recall Curve (AUPRC) and Normalized AUPRC

    Normalized AUPRC is calculated by dividing the AUPRC by the fraction of
    positive class samples in the total samples. This helps adjust for class imbalance.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities or scores

    Returns:
        Dictionary containing both AUPRC and normalized AUPRC
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check if more than one class is present
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true. Returning NaN")
        return {"auprc": np.nan, "normalized_auprc": np.nan}

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
    normalize: bool = False,
) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC) (-1: total disagreement,
    0: random prediction, 1: perfect prediction)

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

    mcc = matthews_corrcoef(y_true, y_pred_binary)
    if normalize:
        mcc = (mcc + 1) / 2

    return mcc


def calculate_kappa(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> float:
    """
    Calculate Cohen's Kappa score (Accounts for agreement due to chance).
    Formula: kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities.
        threshold: Threshold to convert probabilities to binary predictions.
            Defaults to 0.5.

    Returns:
        The Cohen's Kappa score, ranging from -1 to 1:
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    return cohen_kappa_score(y_true, y_pred_binary)


def calculate_minpse(
    y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate MinPSE (Minimum of Precision and Sensitivity Everywhere) score.
    Adapted from Zhu et al. EMERGE (2024) and ColaCare (2025) papers.

    MinPSE finds the threshold where the minimum of precision and recall is maximized,
    providing a balanced operating point for imbalanced classification problems.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities.

    Returns:
        The MinPSE score, ranging from 0 to 1, with higher values indicating better
        balanced performance.
    """
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate precision-recall curve
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)

    # For each threshold point, take the minimum of precision and recall
    # Then find the maximum of these minimums
    minpse_score = np.max([min(p, r) for p, r in zip(precisions, recalls)])

    return minpse_score


def calculate_all_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold=0.5,
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
    # Handle tensors if passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Filter out NaN values in y_true and y_pred
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]
    # Log the number of filtered values (NaNs)
    num_filtered = np.size(y_true) - np.count_nonzero(valid_indices)
    if num_filtered > 0:
        logger.info(f"Filtered out {num_filtered} NaN value(s) from y_true/y_pred.")

    # Auto-detect if predictions are logits or probabilities
    if np.any((y_pred < 0) | (y_pred > 1)):
        # Convert logits to probabilities using sigmoid
        y_pred = 1 / (1 + np.exp(-y_pred))

    # Get both AUPRC and normalized AUPRC in one call
    auprc_results = calculate_auprc(y_true, y_pred)

    metrics = {
        "auroc": calculate_auroc(y_true, y_pred),
        "auprc": auprc_results["auprc"],
        "normalized_auprc": auprc_results["normalized_auprc"],
        "specificity": calculate_specificity(y_true, y_pred, threshold),
        "f1_score": calculate_f1_score(y_true, y_pred, threshold),
        "accuracy": calculate_accuracy(y_true, y_pred, threshold),
        "balanced_accuracy": calculate_balanced_accuracy(y_true, y_pred, threshold),
        "precision": calculate_precision(y_true, y_pred, threshold),
        "recall": calculate_recall(
            y_true, y_pred, threshold
        ),  # is the same as sensitivity
        "mcc": calculate_mcc(y_true, y_pred, threshold),
        "kappa": calculate_kappa(y_true, y_pred, threshold),
        "minpse": calculate_minpse(y_true, y_pred),
    }

    # Round all metrics to 3 decimal places
    rounded_metrics = {
        k: round(v, 3) if not np.isnan(v) else v for k, v in metrics.items()
    }

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
