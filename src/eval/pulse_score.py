import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    matthews_corrcoef,
)
from typing import Dict, List, Optional
import warnings


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
        self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
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
        # Calculate AUPRC
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)

        # Calculate AUROC
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Handle case where only one class is present
            auroc = 0.5
            warnings.warn("Only one class present in y_true. AUROC set to 0.5")

        # Calculate MCC (handle NaN predictions)
        try:
            # Remove NaN predictions for MCC calculation
            valid_mask = ~np.isnan(y_pred)
            if np.any(valid_mask):
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask].astype(int)
                mcc = matthews_corrcoef(y_true_valid, y_pred_valid)
            else:
                # All predictions are NaN - worst possible MCC
                mcc = -1.0
        except ValueError:
            # Handle edge cases in MCC calculation
            mcc = -1.0
            warnings.warn("Could not calculate MCC. Set to -1.0")

        # Normalize MCC from [-1, 1] to [0, 1] for consistency
        mcc_normalized = (mcc + 1) / 2

        return {"auprc": auprc, "auroc": auroc, "mcc": mcc_normalized, "mcc_raw": mcc}

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
        self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate PULSE score for a single outcome.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            y_pred: Binary predictions

        Returns:
            Dictionary containing all metrics and final PULSE score
        """
        # Calculate base metrics
        base_metrics = self.calculate_base_metrics(y_true, y_prob, y_pred)

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
                condition_mentioned = task_lower in pred_lower

                # Check if it's a negation
                is_negation = (
                    "not-" in pred_lower
                    or "not " in pred_lower
                    or "no " in pred_lower
                    or "negative" in pred_lower
                )

                # If the condition is not mentioned at all, return NaN
                if not condition_mentioned:
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
                "predicted_probability",
                "probability",
                "confidence",
                "logits",
                "scores",
                "outputs",
                "Predicted Probability",
                "Logits",
                "Scores",
            ]
        if pred_col_candidates is None:
            pred_col_candidates = [
                "predicted_label",
                "prediction",
                "predicted_diagnosis",
                "logits",
                "pred",
                "preds",
                "labels",
                "Predicted Diagnosis",
                "Prediction",
                "Pred",
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
                    df_prepared["Predicted Diagnosis"] = (
                        df_prepared[prediction_col] > 0
                    ).astype(float)
                else:
                    # Convert various prediction formats to binary with task awareness
                    # Use a lambda with default argument to avoid variable capture issues
                    df_prepared["Predicted Diagnosis"] = df_prepared.apply(
                        lambda row, col=prediction_col: self.convert_prediction_to_binary(
                            row[col], row["task"]
                        ),
                        axis=1,
                    )
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
                df_prepared["Predicted Diagnosis"] = (df_prepared[prob_col] > 0).astype(
                    float
                )
            else:
                # For probabilities, use 0.5 as threshold
                df_prepared["Predicted Diagnosis"] = (
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
        df_prepared = df_prepared.dropna(how="any")

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
            pred_unique = df["Predicted Diagnosis"].dropna().unique()
            results["pred_is_binary"] = all(val in [0, 1] for val in pred_unique)

            # Count NaN predictions for reporting
            nan_count = df["Predicted Diagnosis"].isna().sum()
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
        print("=" * 50)
        print("DATA VERIFICATION FOR PULSE SCORE")
        print("=" * 50)
        print(f"Model Type: {'LLM' if self.is_llm_model else 'Conventional ML'}")

        # Check results
        print(
            f"✓ Required columns present: {verification_results['has_required_columns']}"
        )
        print(f"✓ Target Label is binary: {verification_results['target_is_binary']}")
        print(
            f"✓ Predicted Diagnosis is binary: {verification_results['pred_is_binary']}"
        )
        print(
            f"✓ Predicted Probability in [0,1]: {verification_results['prob_in_range']}"
        )

        # Show NaN predictions if any
        if (
            "nan_predictions" in verification_results
            and verification_results["nan_predictions"] > 0
        ):
            print(
                f"⚠️  Invalid predictions (NaN): {verification_results['nan_predictions']}"
            )
        else:
            print("✓ No invalid predictions detected")

        if verification_results["ready_for_pulse"]:
            print("\n✅ Data format is correct for PULSE calculation!")

            # Show distribution by task
            print("\nData distribution by task:")
            for task in df["task"].unique():
                task_data = df[df["task"] == task]
                print(f"\n{task.upper()}:")
                print(f"  Total samples: {len(task_data)}")
                print(f"  Positive labels: {task_data['Target Label'].sum()}")
                valid_preds = task_data["Predicted Diagnosis"].dropna()
                print(f"  Valid predictions: {len(valid_preds)}")
                print(f"  Positive predictions: {valid_preds.sum()}")
                nan_preds = task_data["Predicted Diagnosis"].isna().sum()
                if nan_preds > 0:
                    print(f"  Invalid predictions (NaN): {nan_preds}")

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
        pred_col: str = "Predicted Diagnosis",
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
        results = {}
        task_scores = {}
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

                # Store result with combined key
                combo_key = f"{task}_{dataset}"
                task_dataset_scores[combo_key] = task_dataset_result["pulse_score"]
                results[combo_key] = task_dataset_result

        # Calculate aggregated scores for each task (mean of task-dataset combinations)
        for task in df[task_col].unique():
            # Get all task-dataset scores for this task
            task_dataset_keys = [
                key for key in task_dataset_scores.keys() if key.startswith(f"{task}_")
            ]

            if task_dataset_keys:
                # Calculate mean score across datasets for this task
                task_scores_list = [
                    task_dataset_scores[key] for key in task_dataset_keys
                ]
                aggregated_score = np.mean(task_scores_list)
                task_scores[task] = aggregated_score

                # Create aggregated result dictionary
                results[task] = {
                    "pulse_score": aggregated_score,
                    "dataset_scores": {
                        key.split("_", 1)[1]: task_dataset_scores[key]
                        for key in task_dataset_keys
                    },
                    "num_datasets": len(task_dataset_keys),
                }

        # Calculate weighted overall score
        if task_scores:
            total_weight = 0
            weighted_sum = 0

            for task, score in task_scores.items():
                weight = self.outcome_weights.get(task, 1.0 / len(task_scores))
                weighted_sum += weight * score
                total_weight += weight

            overall_score = weighted_sum / total_weight if total_weight > 0 else 0

            results["overall"] = {
                "pulse_score": overall_score,
                "task_scores": task_scores,
                "task_dataset_scores": task_dataset_scores,
                "weights_used": self.outcome_weights,
            }

        return results

    def get_score_interpretation(self, score: float) -> str:
        """
        Get interpretation of PULSE score.

        Args:
            score: PULSE score (0-100)

        Returns:
            String interpretation of the score
        """
        if score >= 90:
            return "Excellent performance with optimal confidence calibration"
        elif score >= 75:
            return "Very good performance with good confidence calibration"
        elif score >= 60:
            return "Good performance with acceptable confidence calibration"
        elif score >= 40:
            return "Moderate performance with confidence issues"
        elif score >= 20:
            return "Poor performance with significant confidence problems"
        else:
            return "Very poor performance requiring major improvements"

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
                            "MCC (normalized)": result["mcc"],
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
                        "MCC (normalized)": None,
                        "CCF": None,
                        "Confidence Penalty": None,
                    }
                )

        return pd.DataFrame(comparison_data)

    def plot_pulse_analysis(
        self, pulse_results: Dict[str, Dict[str, float]], model_name: str = "Model"
    ) -> None:
        """
        Create comprehensive PULSE score analysis plots.

        Args:
            pulse_results: Results from PULSE score calculation
            model_name: Name of the model for plot titles
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plots.")
            return

        df_comparison = self.create_pulse_comparison_dataframe(pulse_results)

        if df_comparison.empty:
            print("No data available for plotting")
            return

        # Create the plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"{model_name} - PULSE Score Analysis", fontsize=16, fontweight="bold"
        )

        tasks = df_comparison["Task"]
        pulse_scores = df_comparison["PULSE Score"]
        base_scores = df_comparison["Base Score (no penalties)"]

        # Plot 1: PULSE vs Base Score
        axes[0, 0].bar(
            tasks,
            base_scores,
            alpha=0.7,
            label="Base Score (no penalties)",
            color="lightblue",
        )
        axes[0, 0].bar(
            tasks,
            pulse_scores,
            alpha=0.8,
            label="PULSE Score (with penalties)",
            color="darkblue",
        )
        axes[0, 0].set_title("PULSE Score vs Base Score")
        axes[0, 0].set_ylabel("Score (0-100)")
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Plot 2: Individual Metrics
        x_pos = np.arange(len(tasks))
        width = 0.25

        axes[0, 1].bar(
            x_pos - width, df_comparison["AUPRC"], width, label="AUPRC", alpha=0.8
        )
        axes[0, 1].bar(x_pos, df_comparison["AUROC"], width, label="AUROC", alpha=0.8)
        axes[0, 1].bar(
            x_pos + width,
            df_comparison["MCC (normalized)"],
            width,
            label="MCC (norm)",
            alpha=0.8,
        )
        axes[0, 1].set_title("Individual Metrics by Task")
        axes[0, 1].set_ylabel("Score (0-1)")
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(tasks, rotation=45)
        axes[0, 1].legend()

        # Plot 3: Confidence Penalties
        axes[1, 0].bar(
            tasks, df_comparison["Confidence Penalty"], color="red", alpha=0.7
        )
        axes[1, 0].set_title("Average Consistency Penalty by Task")
        axes[1, 0].set_ylabel("Penalty (0-0.5)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: CCF Impact
        axes[1, 1].bar(tasks, df_comparison["CCF"], color="green", alpha=0.7)
        axes[1, 1].set_title("Confidence-Correctness Factor (CCF)")
        axes[1, 1].set_ylabel("CCF (0-1)")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].axhline(
            y=1.0, color="black", linestyle="--", alpha=0.5, label="Perfect CCF"
        )
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def print_detailed_report(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Print a detailed report of PULSE scores.

        Args:
            results: Results dictionary from calculate_pulse_score_from_dataframe
        """
        print("=" * 60)
        print("PULSE SCORE DETAILED REPORT")
        print("=" * 60)
        print(f"Model Type: {'LLM' if self.is_llm_model else 'Conventional ML'}")
        print(
            f"Weights: AUPRC={self.alpha:.2f}, AUROC={self.beta:.2f}, MCC={self.mcc_weight:.2f}"
        )
        print()

        # Separate task-level and task-dataset-level results
        task_results = {}
        task_dataset_results = {}

        for key, result in results.items():
            if key == "overall":
                continue
            elif "_" in key:
                task_dataset_results[key] = result
            else:
                task_results[key] = result

        # Print results for each task (aggregated)
        if task_results:
            print("TASK-LEVEL SCORES (Mean across datasets)")
            print("=" * 50)
            for task, result in task_results.items():
                print(f"Task: {task.upper()}")
                print("-" * 30)
                print(f"PULSE Score (Mean): {result['pulse_score']:.2f}")
                print(
                    f"Interpretation: {self.get_score_interpretation(result['pulse_score'])}"
                )
                print(f"Number of datasets: {result['num_datasets']}")
                print(
                    f"Dataset scores: {', '.join([f'{ds}={score:.1f}' for ds, score in result['dataset_scores'].items()])}"
                )
                print()

        # Print results for each task-dataset combination
        if task_dataset_results:
            print("TASK-DATASET COMBINATION SCORES")
            print("=" * 50)
            # Group by task for better organization
            task_groups = {}
            for key, result in task_dataset_results.items():
                task, dataset = key.split("_", 1)
                if task not in task_groups:
                    task_groups[task] = {}
                task_groups[task][dataset] = result

            for task, datasets in task_groups.items():
                print(f"\n{task.upper()} by Dataset:")
                print("-" * 40)
                for dataset, result in datasets.items():
                    print(f"  Dataset: {dataset}")
                    print(f"    PULSE Score: {result['pulse_score']:.2f}")
                    print(f"    Base Score: {result['base_score']:.3f}")
                    print(f"    CCF: {result['ccf']:.3f}")
                    print(f"    AUPRC: {result['auprc']:.3f}")
                    print(f"    AUROC: {result['auroc']:.3f}")
                    print(f"    MCC (norm): {result['mcc']:.3f}")
                    if self.is_llm_model and result.get("num_penalized", 0) > 0:
                        print(f"    Problematic Predictions: {result['num_penalized']}")
                    print()

        # Print overall score
        if "overall" in results:
            overall = results["overall"]
            print("OVERALL PULSE SCORE")
            print("=" * 30)
            print(f"Score: {overall['pulse_score']:.2f}")
            print(
                f"Interpretation: {self.get_score_interpretation(overall['pulse_score'])}"
            )
            print(f"Task Weights: {overall['weights_used']}")
            print()

    def print_pulse_insights_report(
        self, pulse_results: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Print detailed insights and recommendations based on PULSE results.

        Args:
            pulse_results: Results from PULSE score calculation
        """
        if "overall" not in pulse_results:
            print("❌ No overall PULSE score available")
            return

        print("=" * 80)
        print("🔍 KEY INSIGHTS - PULSE SCORE ANALYSIS")
        print("=" * 80)

        overall_score = pulse_results["overall"]["pulse_score"]

        print(f"🎯 FINAL PULSE SCORE: {overall_score:.2f}/100")
        print(
            f"📈 Performance Category: {self.get_score_interpretation(overall_score)}"
        )

        # Calculate impact of confidence penalties - only for task-dataset combinations
        task_dataset_results = {
            k: v
            for k, v in pulse_results.items()
            if k != "overall" and "_" in k and "base_score" in v
        }

        if task_dataset_results:
            total_penalty_impact = 0
            total_base_score = 0

            for task_dataset, result in task_dataset_results.items():
                base_without_penalty = result["base_score"] * 100
                penalty_reduction = base_without_penalty - result["pulse_score"]
                total_penalty_impact += penalty_reduction
                total_base_score += base_without_penalty

                print(f"\n📊 {task_dataset.upper()} Analysis:")
                print(f"   • Base performance: {base_without_penalty:.1f}/100")
                print(f"   • Final PULSE score: {result['pulse_score']:.1f}/100")
                print(f"   • Penalty impact: -{penalty_reduction:.1f} points")
                if self.is_llm_model:
                    print(
                        f"   • Problematic predictions: {result['num_penalized']} (inconsistent or invalid)"
                    )

            if self.is_llm_model and task_dataset_results:
                avg_penalty_impact = total_penalty_impact / len(task_dataset_results)
                print("\n⚠️  PREDICTION QUALITY:")
                print(
                    f"   • Average penalty impact: -{avg_penalty_impact:.1f} points per task-dataset"
                )
                severity = (
                    "significant"
                    if avg_penalty_impact > 10
                    else "moderate" if avg_penalty_impact > 5 else "minimal"
                )
                print(f"   • This indicates {severity} issues with prediction quality")
                print(
                    "   • Issues include: confidence-prediction inconsistency & invalid task understanding"
                )

        # Show aggregated task scores
        task_results = {
            k: v for k, v in pulse_results.items() if k != "overall" and "_" not in k
        }

        if task_results:
            print("\n📈 AGGREGATED TASK PERFORMANCE:")
            for task, result in task_results.items():
                print(
                    f"   • {task.capitalize()}: {result['pulse_score']:.1f}/100 (mean across {result['num_datasets']} datasets)"
                )

        print("\n🔬 CLINICAL IMPACT:")
        for task in ["sepsis", "mortality", "aki"]:
            score = pulse_results.get(task, {}).get("pulse_score", 0)
            print(f"   • {task.capitalize()} prediction: {score:.1f}/100")

    def calculate_pulse_score_from_raw_data(
        self,
        df: pd.DataFrame,
        model_name: str = "Model",
        target_col_candidates: List[str] = None,
        prob_col_candidates: List[str] = None,
        pred_col_candidates: List[str] = None,
        task_col_candidates: List[str] = None,
        dataset_col_candidates: List[str] = None,
        show_verification: bool = True,
        show_plots: bool = True,
        show_insights: bool = True,
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
            show_verification: Whether to show data verification report
            show_plots: Whether to show analysis plots
            show_insights: Whether to show insights report
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
                pred_col="Predicted Diagnosis",
                task_col="task",
                dataset_col="dataset",
            )

            print(f"✅ PULSE scores calculated successfully for {model_name}!")

            # Step 4: Generate reports and visualizations
            if show_detailed_report:
                self.print_detailed_report(pulse_results)

            if show_plots:
                self.plot_pulse_analysis(pulse_results, model_name)

            if show_insights:
                self.print_pulse_insights_report(pulse_results)

            return pulse_results

        except Exception as e:
            print(f"❌ Error calculating PULSE scores for {model_name}: {e}")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"DataFrame shape: {df.shape}")
            raise e
