import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.eval.metrics import (calculate_accuracy, calculate_auprc,
                              calculate_auroc, calculate_balanced_accuracy,
                              calculate_f1_score, calculate_kappa,
                              calculate_mcc, calculate_minpse,
                              calculate_precision, calculate_recall,
                              calculate_specificity)


class PULSEScoreCalculator:
    """
    PULSE Score Calculator for PULSE Models.

    The PULSE score combines traditional ML metrics (AUPRC, AUROC, MCC) with a confidence-correctness
    factor (CCF) that penalizes LLMs for inconsistency between confidence and predictions.

    Formula:
    PULSE_outcome = 100 × (α·AUPRC + β·AUROC + (1-α-β)·MCC) × CCF
    PULSE_total = Σ(γ_j · PULSE_outcome_j)

    Features:
    - Automatic data preparation and column mapping
    - Support for both LLM and conventional ML models
    - Comprehensive reporting and visualization
    - Data format verification
    - Detailed insights and recommendations

    CCF Calculation (for LLMs):
    - Penalizes cases where explicit prediction doesn't match binarized probability
    - Penalizes invalid predictions (NaN) that don't match the expected task format
    - Encourages consistency between confidence (probability) and prediction
    - Encourages proper task understanding (e.g., "sepsis" vs "not-sepsis" for sepsis task)
    """

    def __init__(
        self,
        alpha: float = 0.33,
        beta: float = 0.33,
        outcome_weights: Optional[Dict[str, float]] = None,
        is_llm_model: bool = True,
    ):
        """
        Initialize the PULSE score calculator.

        Args:
            alpha: Weight for AUPRC in base score (default: 0.33)
            beta: Weight for AUROC in base score (default: 0.33)
            outcome_weights: Clinical importance weights for each outcome/task
            is_llm_model: Whether the model is an LLM (affects CCF calculation)
        """
        self.alpha = alpha
        self.beta = beta
        self.mcc_weight = 1 - alpha - beta
        self.is_llm_model = is_llm_model

        # Validate weights
        if not np.isclose(alpha + beta + self.mcc_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
        if any(w < 0 for w in [alpha, beta, self.mcc_weight]):
            raise ValueError("All weights must be non-negative")

        # Default outcome weights (can be customized)
        self.outcome_weights = outcome_weights or {
            "sepsis": 0.33,
            "mortality": 0.33,
            "aki": 0.33,
        }

    def calculate_base_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate the three base metrics: AUPRC, AUROC, and MCC.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            y_pred: Binary predictions

        Returns:
            Dictionary containing AUPRC, AUROC, and MCC scores
        """
        auprc_results = calculate_auprc(y_true, y_prob)
        threshold = 0.5
        metrics = {
            "auroc": calculate_auroc(y_true, y_prob),
            "auprc": auprc_results["auprc"],
            "mcc_raw": calculate_mcc(y_true, y_prob, threshold, normalize=False),
            "mcc": calculate_mcc(y_true, y_prob, threshold, normalize=True),
            "auroc": calculate_auroc(y_true, y_prob),
            "normalized_auprc": auprc_results["normalized_auprc"],
            "specificity": calculate_specificity(y_true, y_prob, threshold),
            "f1_score": calculate_f1_score(y_true, y_prob, threshold),
            "accuracy": calculate_accuracy(y_true, y_prob, threshold),
            "balanced_accuracy": calculate_balanced_accuracy(y_true, y_prob, threshold),
            "precision": calculate_precision(y_true, y_prob, threshold),
            "recall": calculate_recall(
                y_true, y_prob, threshold
            ),  # is the same as sensitivity
            "kappa": calculate_kappa(y_true, y_prob, threshold),
            "minpse": calculate_minpse(y_true, y_prob),
        }

        return metrics

    def calculate_ccf(
        self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate the Confidence-Correctness Factor (CCF).

        For LLM models, this penalizes:
        1. Inconsistency between predicted probability and explicit prediction
        2. Invalid predictions that don't match the task (e.g., "not-diagnosis" for sepsis task)

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            y_pred: Binary predictions (may contain NaN for invalid predictions)

        Returns:
            Dictionary containing CCF and penalty statistics
        """
        penalties = np.zeros(len(y_true))

        if self.is_llm_model:
            # Handle NaN predictions (invalid/improper understanding)
            valid_mask = ~np.isnan(y_pred)

            if not np.any(valid_mask):
                # All predictions are invalid - maximum penalty
                avg_penalty = 0.5
                ccf = 1 - avg_penalty
                return {
                    "ccf": ccf,
                    "avg_penalty": avg_penalty,
                    "total_penalties": len(y_true) * 0.5,
                    "num_penalized": len(y_true),
                }

            # Only consider valid predictions for consistency check
            y_prob_valid = y_prob[valid_mask]
            y_pred_valid = y_pred[valid_mask].astype(int)

            # Calculate penalties when prediction doesn't match binarized probability
            # This penalizes inconsistency between confidence (probability) and prediction
            y_prob_binarized = (y_prob_valid > 0.5).astype(int)
            inconsistent_mask = y_pred_valid != y_prob_binarized

            # Apply penalties to valid predictions
            valid_penalties = np.zeros(len(y_prob_valid))
            valid_penalties[inconsistent_mask] = np.abs(
                y_prob_valid[inconsistent_mask] - 0.5
            )

            # Assign maximum penalty (0.5) to invalid predictions (NaN)
            all_penalties = np.full(len(y_true), 0.0)
            all_penalties[valid_mask] = valid_penalties
            all_penalties[~valid_mask] = 0.5  # Maximum penalty for NaN predictions

            penalties = all_penalties

        # CCF is 1 minus average penalty
        avg_penalty = np.mean(penalties)
        ccf = 1 - avg_penalty

        return {
            "ccf": ccf,
            "avg_penalty": avg_penalty,
            "total_penalties": np.sum(penalties),
            "num_penalized": np.sum(penalties > 0),
        }

    def calculate_base_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate the base score using weighted combination of metrics.

        Args:
            metrics: Dictionary containing auprc, auroc, and mcc scores

        Returns:
            Base score (0-1 range)
        """
        base_score = (
            self.alpha * metrics["auprc"]
            + self.beta * metrics["auroc"]
            + self.mcc_weight * metrics["mcc"]
        )
        return base_score

    def calculate_pulse_score_single_outcome(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        base_metrics=None,
    ) -> Dict[str, float]:
        """
        Calculate PULSE score for a single outcome.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            y_pred: Binary predictions
            base_metrics: Pre-calculated base metrics (optional)

        Returns:
            Dictionary containing all metrics and final PULSE score
        """
        # Calculate base metrics if not given
        if base_metrics is None:
            base_metrics = self.calculate_base_metrics(y_true, y_prob)

        # Calculate base score
        base_score = self.calculate_base_score(base_metrics)

        # Calculate CCF
        ccf_metrics = self.calculate_ccf(y_true, y_prob, y_pred)

        # Calculate final PULSE score (0-100 scale)
        pulse_score = 100 * base_score * ccf_metrics["ccf"]

        # Combine all results
        result = {
            "pulse_score": pulse_score,
            "base_score": base_score,
            **base_metrics,
            **ccf_metrics,
        }

        return result

    def convert_prediction_to_binary(self, pred_value, task: str = None) -> float:
        """
        Convert various prediction formats to binary format, with task-aware validation.

        Args:
            pred_value: Prediction string, logit, probability, or binary value
            task: The task name (sepsis, aki, mortality) for validation

        Returns:
            Binary prediction (0 or 1) or NaN if prediction doesn't match task
        """
        if isinstance(pred_value, str):
            pred_lower = pred_value.lower()

            # Task-aware validation for LLM string predictions
            if task is not None:
                task_lower = task.lower()

                # Check if prediction mentions the correct condition
                conditions = [
                    task_lower,
                    f"not-{task_lower}",
                    "diagnosis",
                    "not-diagnosis",
                ]  # also allowing to say diagnosis or not-diagnosis

                # Check if it's a negation
                is_negation = (
                    "not-" in pred_lower
                    or "not " in pred_lower
                    or "no " in pred_lower
                    or "negative" in pred_lower
                )

                # If the condition is not mentioned at all, return NaN
                if pred_lower not in conditions:
                    return np.nan

                # If condition is mentioned, determine positive/negative
                if is_negation:
                    return 0.0
                else:
                    return 1.0
            else:
                # Fallback for when no task is provided
                if "not-" in pred_lower or "not " in pred_lower:
                    return 0.0
                else:
                    return 1.0

        elif isinstance(pred_value, (int, float)):
            # Handle numeric values (logits, probabilities, or binary)
            if np.isnan(pred_value):
                return np.nan
            elif pred_value in [0, 1]:
                return float(pred_value)
            elif pred_value > 1 or pred_value < 0:
                # Treat as logits - convert using sigmoid-like threshold
                return 1.0 if pred_value > 0 else 0.0
            else:
                # Treat as probability
                return 1.0 if pred_value > 0.5 else 0.0
        else:
            # Unknown format - return NaN
            return np.nan

    def convert_logits_to_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """
        Convert logits to probabilities using sigmoid function.

        Args:
            logits: Array of logit values

        Returns:
            Array of probabilities between 0 and 1
        """
        return 1 / (1 + np.exp(-logits))

    def prepare_dataframe_for_pulse(
        self,
        df: pd.DataFrame,
        target_col_candidates: List[str] = None,
        prob_col_candidates: List[str] = None,
        pred_col_candidates: List[str] = None,
        task_col_candidates: List[str] = None,
        dataset_col_candidates: List[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare a DataFrame for PULSE score calculation by mapping and converting columns.

        Args:
            df: Input DataFrame
            target_col_candidates: List of possible target/label column names
            prob_col_candidates: List of possible probability/logits column names
            pred_col_candidates: List of possible prediction column names (optional for conventional models)
            task_col_candidates: List of possible task column names
            dataset_col_candidates: List of possible dataset column names

        Returns:
            DataFrame with standardized column names for PULSE calculation

        Note:
            - For LLM models: Prediction columns are preferred but can be derived from probabilities
            - For conventional models: Predictions are typically derived from probabilities/logits
            - Logits are automatically converted to probabilities using sigmoid function
        """
        # Default candidate column names
        if target_col_candidates is None:
            target_col_candidates = ["true_label", "label", "target", "Target Label"]
        if prob_col_candidates is None:
            prob_col_candidates = [
                "Predicted Probability",
            ]
        if pred_col_candidates is None:
            pred_col_candidates = [
                "Predicted Diagnosis",
            ]
        if task_col_candidates is None:
            task_col_candidates = ["task", "task_id", "outcome"]
        if dataset_col_candidates is None:
            dataset_col_candidates = ["dataset", "split", "data_split"]

        df_prepared = df.copy()

        # Map target/label column
        target_col = None
        for col in target_col_candidates:
            if col in df_prepared.columns:
                df_prepared["Target Label"] = df_prepared[col].astype(int)
                target_col = col
                break

        if target_col is None:
            raise ValueError(
                f"No target column found. Looked for: {target_col_candidates}"
            )

        # Map probability column
        prob_col = None
        prob_is_logits = False
        for col in prob_col_candidates:
            if col in df_prepared.columns:
                if "logit" in col.lower():
                    # Convert logits to probabilities
                    df_prepared["Predicted Probability"] = (
                        self.convert_logits_to_probabilities(df_prepared[col].values)
                    )
                    prob_is_logits = True
                else:
                    df_prepared["Predicted Probability"] = df_prepared[col]
                prob_col = col
                break

        if prob_col is None:
            raise ValueError(
                f"No probability/logits column found. Looked for: {prob_col_candidates}"
            )

        # Map task column first (we need this for task-aware prediction conversion)
        task_col = None
        for col in task_col_candidates:
            if col in df_prepared.columns:
                df_prepared["task"] = df_prepared[col]
                task_col = col
                break

        if task_col is None:
            raise ValueError(f"No task column found. Looked for: {task_col_candidates}")

        # Map prediction column - more flexible approach with task awareness
        pred_col = None

        # First, try to find explicit prediction columns
        for prediction_col in pred_col_candidates:
            if prediction_col in df_prepared.columns:
                if "logit" in prediction_col.lower():
                    # Convert logits to binary predictions
                    df_prepared["Predicted Diagnosis Binary"] = (
                        df_prepared[prediction_col] > 0
                    ).astype(float)
                else:
                    # Convert various prediction formats to binary with task awareness
                    # If task label is not found in prediction, it will return NaN
                    # Use a lambda with default argument to avoid variable capture issues
                    df_prepared["Predicted Diagnosis Binary"] = df_prepared.apply(
                        lambda row, col=prediction_col: self.convert_prediction_to_binary(
                            row[col], row["task"]
                        ),
                        axis=1,
                    )
                    # Convert Probability Prediction to Binary
                    df_prepared["Predicted Binary"] = (
                        df_prepared["Predicted Probability"] > 0.5
                    ).astype(float)

                pred_col = prediction_col
                break

        # If no explicit prediction column found, derive from probabilities
        if pred_col is None:
            if self.is_llm_model:
                print(
                    "Warning: No prediction column found for LLM. Deriving from probabilities (>0.5)"
                )
            else:
                print(
                    "Info: No explicit prediction column found for conventional model. Deriving from probabilities/logits (>0.5 or >0 for logits)"
                )

            if prob_is_logits:
                # For logits, use 0 as threshold
                df_prepared["Predicted Binary"] = (df_prepared[prob_col] > 0).astype(
                    float
                )
            else:
                # For probabilities, use 0.5 as threshold
                df_prepared["Predicted Binary"] = (
                    df_prepared["Predicted Probability"] > 0.5
                ).astype(float)

        # Map dataset column
        dataset_col = None
        for col in dataset_col_candidates:
            if col in df_prepared.columns:
                df_prepared["dataset"] = df_prepared[col]
                dataset_col = col
                break

        if dataset_col is None:
            print("Warning: No dataset column found. Using 'test' as default.")
            df_prepared["dataset"] = "test"

        verification_results = self.verify_pulse_data_format(df_prepared)
        if verbose:
            self.print_data_verification_report(df_prepared, verification_results)

        if not verification_results["ready_for_pulse"]:
            raise ValueError(
                "Data format verification failed. Please check the data preparation."
            )

        # Remove cols with full NaN
        df_prepared = df_prepared.dropna(axis=1, how="all")
        # Remove rows with NaN values for robustness
        # df_prepared = df_prepared.dropna(how="any")
        df_prepared = df_prepared.dropna(subset=["Predicted Probability"])

        return df_prepared

    def verify_pulse_data_format(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Verify that DataFrame is properly formatted for PULSE calculation.

        Args:
            df: DataFrame to verify

        Returns:
            Dictionary with verification results
        """
        results = {
            "has_required_columns": False,
            "target_is_binary": False,
            "pred_is_binary": False,
            "prob_in_range": False,
            "ready_for_pulse": False,
        }

        required_cols = [
            "Target Label",
            "Predicted Probability",
            "Predicted Diagnosis",
            "Predicted Binary",
            "task",
            "dataset",
        ]

        # Check required columns
        if all(col in df.columns for col in required_cols):
            results["has_required_columns"] = True

            # Check target label format
            target_unique = df["Target Label"].unique()
            results["target_is_binary"] = all(val in [0, 1] for val in target_unique)

            # Check prediction format (allow NaN for invalid predictions)
            pred_unique = df["Predicted Binary"].dropna().unique()
            results["pred_is_binary"] = all(val in [0, 1] for val in pred_unique)

            # Count NaN predictions for Diagnosis. e.g. is nan when diagnosis is not correctly labeled as asked in prompt
            nan_count = df["Predicted Diagnosis Binary"].isna().sum()
            results["nan_predictions"] = nan_count

            # Check probability range
            prob_min = df["Predicted Probability"].min()
            prob_max = df["Predicted Probability"].max()
            results["prob_in_range"] = prob_min >= 0 and prob_max <= 1

            # Overall readiness
            results["ready_for_pulse"] = all(
                [
                    results["target_is_binary"],
                    results["pred_is_binary"],
                    results["prob_in_range"],
                ]
            )

        return results

    def print_data_verification_report(
        self, df: pd.DataFrame, verification_results: Dict[str, bool]
    ) -> None:
        """
        Print a detailed data verification report.

        Args:
            df: DataFrame being verified
            verification_results: Results from verify_pulse_data_format
        """
        print(f"Model Type: {'LLM' if self.is_llm_model else 'Conventional ML'}")

        # Show NaN predictions if any
        if (
            "nan_predictions" in verification_results
            and verification_results["nan_predictions"] > 0
        ):
            print(
                f"\n⚠️ Wrongly labled predictions: {verification_results['nan_predictions']}"
            )
            wrongly_labeled = df[df["Predicted Diagnosis Binary"].isna()]
            if not wrongly_labeled.empty:
                value_counts = wrongly_labeled["Predicted Diagnosis"].value_counts()
                print("Wrongly labeled predictions:")
                for label, count in value_counts.items():
                    print(f"  - {label}: {count} occurrences")

        else:
            print("✓ No invalid predictions detected")

        if verification_results["ready_for_pulse"]:
            print("✅ Data format is correct for PULSE calculation!")

            # Show distribution by task
            print("\nData distribution by task:")
            for task in df["task"].unique():
                task_data = df[df["task"] == task]
                print(f"\n{task.upper()}:")
                print(f"  Total samples: {len(task_data)}")
                print(f"  Positive labels: {task_data['Target Label'].sum()}")
                valid_preds = task_data["Predicted Binary"].dropna()
                print(f"  Valid predictions: {len(valid_preds)}")
                print(f"  Positive predictions: {valid_preds.sum()}")
                nan_preds = task_data["Predicted Diagnosis"].isna().sum()
                if nan_preds > 0:
                    print(f"  Wrongly labled: {nan_preds}")

                # Show probability statistics
                prob_stats = task_data["Predicted Probability"].describe()
                print(
                    f"  Probability range: [{prob_stats['min']:.3f}, {prob_stats['max']:.3f}]"
                )
                print(f"  Probability mean: {prob_stats['mean']:.3f}")
        else:
            print("\n❌ Data format needs fixing before PULSE calculation")

        # Show data types and unique values
        if verification_results["has_required_columns"]:
            print("\nData types and unique values:")
            key_cols = ["Target Label", "Predicted Diagnosis", "Predicted Probability"]
            for col in key_cols:
                if col in df.columns:
                    unique_vals = df[col].unique()[:5]  # Show first 5 unique values
                    print(f"  {col}: {df[col].dtype}, values: {unique_vals}")

        # Show data derivation info
        print("\nData source information:")
        if "Predicted Diagnosis" in df.columns:
            # Check if predictions seem to be derived from probabilities
            prob_derived = (
                df["Predicted Diagnosis"]
                == (df["Predicted Probability"] > 0.5).astype(int)
            ).all()
            if prob_derived:
                print("  • Predictions derived from probabilities (threshold = 0.5)")
            else:
                print("  • Predictions from explicit prediction column")

        print(f"  • Total samples: {len(df)}")
        print(f"  • Tasks: {', '.join(df['task'].unique())}")
        print(f"  • Datasets: {', '.join(df['dataset'].unique())}")

    def calculate_pulse_score_from_dataframe(
        self,
        df: pd.DataFrame,
        target_col: str = "Target Label",
        prob_col: str = "Predicted Probability",
        pred_col: str = "Predicted Diagnosis Binary",
        task_col: str = "task",
        dataset_col: str = "dataset",
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate PULSE scores from a pandas DataFrame.

        Args:
            df: DataFrame containing predictions and true labels
            target_col: Column name for true binary labels
            prob_col: Column name for predicted probabilities
            pred_col: Column name for binary predictions
            task_col: Column name for task/outcome identifier
            dataset_col: Column name for dataset identifier

        Returns:
            Dictionary with PULSE scores for each task, each task-dataset combination, and overall score
        """
        results_dict = {
            "pulse_scores": [],
            "task_scores": {"aki": [], "mortality": [], "sepsis": []},
            "dataset_scores": {"eicu": [], "miiv": [], "hirid": []},
            "task_dataset_scores": {},
            "overall": {},
        }
        task_dataset_scores = {}

        # Calculate scores for each task-dataset combination
        for task in df[task_col].unique():
            for dataset in df[dataset_col].unique():
                task_dataset_data = df[
                    (df[task_col] == task) & (df[dataset_col] == dataset)
                ].copy()

                if len(task_dataset_data) == 0:
                    continue

                # Extract arrays
                y_true = task_dataset_data[target_col].values
                y_prob = task_dataset_data[prob_col].values
                y_pred = task_dataset_data[pred_col].values

                # Validate data
                if len(np.unique(y_true)) < 2:
                    warnings.warn(
                        f"Task {task} on dataset {dataset} has only one class. Skipping."
                    )
                    continue

                # Calculate PULSE score for this task-dataset combination
                task_dataset_result = self.calculate_pulse_score_single_outcome(
                    y_true, y_prob, y_pred
                )
                task_dataset_result["dataset"] = dataset
                task_dataset_result["task_id"] = task
                task_dataset_result["model_name"] = df["model_name"].iloc[0]
                task_dataset_result["run_id"] = df["timestamp"].iloc[0]

                # Store result with combined key
                combo_key = f"{task}_{dataset}"
                task_dataset_scores[combo_key] = task_dataset_result["pulse_score"]

                results_dict["pulse_scores"].append(task_dataset_result["pulse_score"])
                results_dict["task_scores"][task].append(
                    task_dataset_result["pulse_score"]
                )
                results_dict["dataset_scores"][dataset].append(
                    task_dataset_result["pulse_score"]
                )
                results_dict["task_dataset_scores"][combo_key] = task_dataset_result

        # Calculate overall PULSE score
        results_dict["overall"]["overall_score"] = np.mean(results_dict["pulse_scores"])
        for task, dataset in zip(df[task_col].unique(), df[dataset_col].unique()):

            # Calculate overall scores for tasks and datasets
            if task in results_dict["task_scores"]:
                results_dict["overall"][f"overall_{task}"] = np.mean(
                    results_dict["task_scores"][task]
                )
            else:
                results_dict["overall"][f"overall_{task}"] = 0.0

            if dataset in results_dict["dataset_scores"]:
                results_dict["overall"][f"overall_{dataset}"] = np.mean(
                    results_dict["dataset_scores"][dataset]
                )
            else:
                results_dict["overall"][f"overall_{dataset}"] = 0.0

        return results_dict

    def create_pulse_comparison_dataframe(
        self, pulse_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Create a comparison DataFrame from PULSE results.

        Args:
            pulse_results: Results from PULSE score calculation

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        for task, result in pulse_results.items():
            if task == "overall":
                continue

            # Handle both aggregated task results and task-dataset combination results
            if "_" in task:
                # Task-dataset combination result (has all metrics)
                if "base_score" in result:
                    base_score_100 = result["base_score"] * 100
                    comparison_data.append(
                        {
                            "Task": task.capitalize(),
                            "PULSE Score": result["pulse_score"],
                            "Base Score (no penalties)": base_score_100,
                            "AUPRC": result["auprc"],
                            "AUROC": result["auroc"],
                            "MCC": result["mcc"],
                            "CCF": result["ccf"],
                            "Confidence Penalty": result["avg_penalty"],
                        }
                    )
            else:
                # Aggregated task result (only has pulse_score)
                comparison_data.append(
                    {
                        "Task": task.capitalize(),
                        "PULSE Score": result["pulse_score"],
                        "Base Score (no penalties)": result[
                            "pulse_score"
                        ],  # Use same as PULSE for aggregated
                        "AUPRC": None,
                        "AUROC": None,
                        "MCC": None,
                        "CCF": None,
                        "Confidence Penalty": None,
                    }
                )

        return pd.DataFrame(comparison_data)

    def print_detailed_report(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Print a detailed report of PULSE scores.

        Args:
            results: Results dictionary from calculate_pulse_score_from_dataframe
        """

        print("PULSE SCORE DETAILED REPORT")
        print(f"Model Type: {'LLM' if self.is_llm_model else 'Conventional ML'}")
        print(
            f"Weights: AUPRC={self.alpha:.2f}, AUROC={self.beta:.2f}, MCC={self.mcc_weight:.2f}"
        )
        print()

        # Print overall results
        if "overall" in results:
            overall = results["overall"]
            print("OVERALL SCORES")
            print("-" * 50)
            # Convert overall results to a DataFrame for display
            overall_df = pd.DataFrame(
                list(overall.items()), columns=["Metric", "Value"]
            )
            print(overall_df)
            print()

        # Print Task Dataset Scores
        if "task_dataset_scores" in results:
            print("TASK DATASET SCORES")
            print("-" * 50)
            task_dataset_df = pd.DataFrame.from_dict(
                results["task_dataset_scores"], orient="index"
            )
            print(task_dataset_df[["pulse_score", "auprc", "auroc", "mcc", "ccf"]])

    def calculate_pulse_score_from_raw_data(
        self,
        df: pd.DataFrame,
        model_name: str = "Model",
        target_col_candidates: List[str] = None,
        prob_col_candidates: List[str] = None,
        pred_col_candidates: List[str] = None,
        task_col_candidates: List[str] = None,
        dataset_col_candidates: List[str] = None,
        show_detailed_report: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Complete PULSE score calculation pipeline from raw data to results.
        Handles both LLM and conventional ML models with flexible data formats.

        Args:
            df: Raw DataFrame with model predictions and labels
            model_name: Name of the model for reporting. Is extracted from df
            target_col_candidates: List of possible target column names
            prob_col_candidates: List of possible probability/logits column names
            pred_col_candidates: List of possible prediction column names (optional for conventional models)
            task_col_candidates: List of possible task column names
            dataset_col_candidates: List of possible dataset column names
            show_detailed_report: Whether to show detailed metrics report

        Returns:
            PULSE score results dictionary

        Note:
            - For conventional ML models: predictions can be derived from probabilities/logits if not provided
            - For LLM models: prediction columns are preferred but can be derived from probabilities
            - Logits are automatically converted to probabilities and binary predictions
        """
        try:
            model_name = (
                df["model_name"].iloc[0] if "model_name" in df.columns else model_name
            )

            # Step 1: Prepare the data
            print(f"Preparing data for {model_name} PULSE score calculation...")
            df_prepared = self.prepare_dataframe_for_pulse(
                df=df,
                target_col_candidates=target_col_candidates,
                prob_col_candidates=prob_col_candidates,
                pred_col_candidates=pred_col_candidates,
                task_col_candidates=task_col_candidates,
                dataset_col_candidates=dataset_col_candidates,
                verbose=show_detailed_report,
            )

            # Step 2: Calculate PULSE scores
            print("Calculating PULSE scores...")
            pulse_results = self.calculate_pulse_score_from_dataframe(
                df_prepared,
                target_col="Target Label",
                prob_col="Predicted Probability",
                pred_col="Predicted Diagnosis Binary",
                task_col="task",
                dataset_col="dataset",
            )

            print(f"✅ PULSE scores calculated successfully for {model_name}!")

            # Step 4: Generate reports and visualizations
            if show_detailed_report:
                self.print_detailed_report(pulse_results)

            return pulse_results

        except Exception as e:
            print(f"❌ Error calculating PULSE scores for {model_name}: {e}")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"DataFrame shape: {df.shape}")
            raise e
