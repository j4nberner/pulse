import ast
import glob
import json
import os
import re

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.table import Table
from matplotlib import gridspec
from IPython.display import HTML, display

from src.eval.metrics import calculate_auprc

VARIABLE_NAMES = {
    "model_name": "Model Name",
    "task_id": "Task ID",
    "task": "Task",
    "dataset": "Dataset",
    "timestamp": "Timestamp",
    "model_id": "Model ID",
    "prompting_id": "Prompting ID",
    "sarvari_2024_aggregation_preprocessor": "Aggregation",
    "zhu_2024b_zero_shot_preprocessor": "Zero-shot",
    "liu_2023_few_shot_preprocessor": "Few-shot",
    "zhu_2024b_one_shot_preprocessor": "One-shot",
    "zhu_2024a_cot_preprocessor": "Chain-of-Thought",
    "zhu_2024c_categorization_summary_agent_preprocessor": "Categorization Summary Agent",
    "aki": "AKI",
    "mortality": "Mortality",
    "sepsis": "Sepsis",
}


COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8E44AD", "#27AE60"]


class ModelAnalyzer:
    """
    A class to analyze the performance of conventional Pulse Models.
    """

    def __init__(self):
        pass

    @staticmethod
    def expand_full_dataset(df, prediction_col="prediction", label_col="label"):
        """
        Expand the full dataset where predictions and labels are batched.

        Args:
            df: DataFrame with batched predictions and labels
            prediction_col: Column name containing prediction strings
            label_col: Column name containing label strings

        Returns:
            pd.DataFrame: Expanded DataFrame with only required columns
        """
        expanded_rows = []

        for idx, row in df.iterrows():
            try:
                # Parse predictions
                pred_str = row[prediction_col]
                clean_pred_str = pred_str.replace("\n", " ").replace("  ", " ")
                # Handle both Python list and NumPy array-like string formats
                clean_pred_str = clean_pred_str.strip()
                if (
                    clean_pred_str.startswith("[")
                    and " " in clean_pred_str
                    and "," not in clean_pred_str
                ):
                    # If the string looks like '[[-1.3 ] [-1.4 ] ...]', remove all interior braces
                    # and convert to a flat Python list string
                    # Remove all occurrences of '][' and replace with ','
                    clean_pred_str = clean_pred_str.replace("][", ",")
                    # Remove all remaining '[' and ']'
                    clean_pred_str = clean_pred_str.replace("[", "").replace("]", "")
                    # Split by whitespace and join with commas
                    clean_pred_str = "[" + ", ".join(clean_pred_str.split()) + "]"
                clean_pred_str = re.sub(r"\]\s+\[", "], [", clean_pred_str)
                parsed_predictions = np.array(
                    ast.literal_eval(clean_pred_str)
                ).flatten()

                # Parse labels if they exist and are in similar format
                labels = None
                if label_col in row and pd.notna(row[label_col]):
                    try:
                        label_str = str(row[label_col]).strip()
                        # If predictions were parsed as a list/array, expect labels to be in similar format
                        if clean_pred_str.startswith("[") and label_str.startswith("["):
                            # Clean up label string similar to prediction string
                            clean_label_str = label_str.replace("\n", " ").replace(
                                "  ", " "
                            )
                            if (
                                clean_label_str.startswith("[")
                                and " " in clean_label_str
                                and "," not in clean_label_str
                            ):
                                clean_label_str = clean_label_str.replace("][", ",")
                                clean_label_str = clean_label_str.replace(
                                    "[", ""
                                ).replace("]", "")
                                clean_label_str = (
                                    "[" + ", ".join(clean_label_str.split()) + "]"
                                )
                            clean_label_str = re.sub(
                                r"\]\s+\[", "], [", clean_label_str
                            )
                            parsed_labels = np.array(
                                ast.literal_eval(clean_label_str)
                            ).flatten()
                            labels = parsed_labels
                        else:
                            # Single label repeated for all predictions
                            single_label = float(label_str)
                            labels = np.full(len(parsed_predictions), single_label)
                    except:
                        # If parsing fails, assume single label for all predictions
                        try:
                            single_label = float(row[label_col])
                            labels = np.full(len(parsed_predictions), single_label)
                        except:
                            labels = None

                # Create expanded rows with only required columns
                for i, pred_value in enumerate(parsed_predictions):
                    new_row = {
                        "batch": idx,
                        "model_name": row.get("model_name", "Unknown"),
                        "task": row.get("task", "Unknown"),
                        "dataset": row.get("dataset", "Unknown"),
                        "timestamp": row.get("timestamp", "Unknown"),
                        "probability": float(
                            1 / (1 + np.exp(-pred_value))
                        ),  # Convert logit to probability
                        "binary_prediction": int(
                            1 / (1 + np.exp(-pred_value)) > 0.5
                        ),  # Binary prediction
                    }

                    # Add label if available
                    if labels is not None and i < len(labels):
                        new_row["label_value"] = float(labels[i])
                    elif label_col in row:
                        # Use original label if available
                        try:
                            new_row["label_value"] = float(row[label_col])
                        except:
                            new_row["label_value"] = None
                    else:
                        new_row["label_value"] = None

                    expanded_rows.append(new_row)

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        if expanded_rows:
            expanded_df = pd.DataFrame(expanded_rows)

            # Ensure we only have the required columns in the specified order
            required_columns = [
                "batch",
                "model_name",
                "task",
                "dataset",
                "timestamp",
                "probability",
                "label_value",
                "binary_prediction",
            ]

            # Keep only columns that exist in the DataFrame
            final_columns = [
                col for col in required_columns if col in expanded_df.columns
            ]
            expanded_df = expanded_df[final_columns]

            return expanded_df.reset_index(drop=True)
        else:
            print("No valid data found to expand")
            return pd.DataFrame()

    @staticmethod
    def categorize_files(outputfolder_path_list: list, verbose: bool = True):
        """
        Categorize files in the output folders into metrics report files, metadata files, and log files.

        Args:
            outputfolder_path_list (list): List of output folder paths.

        Returns:
            dict: A dictionary containing categorized files.
        """
        file_list = []
        for outputfolder_path in outputfolder_path_list:
            # List files in the main folder
            # file_list.extend(glob.glob(os.path.join(outputfolder_path, "*")))
            # List files in all subfolders (recursively)
            for root, dirs, files in os.walk(outputfolder_path):
                for file in files:
                    file_list.append(os.path.join(root, file))

        categorized_files = {
            "metrics_report_files": [f for f in file_list if "metrics_report" in f],
            "metadata_files": [f for f in file_list if "metadata" in f],
            "log_files": [f for f in file_list if "log" in f],
        }

        if verbose:
            print("Metrics Report Files:")
            for file in categorized_files["metrics_report_files"]:
                print(file)
            print("\nMetadata Files:")
            for file in categorized_files["metadata_files"]:
                print(file)
            print("\nLog Files:")
            for file in categorized_files["log_files"]:
                print(file)

        return categorized_files


class LLMAnalyzer(ModelAnalyzer):
    """
    A class to analyze the performance of a Pulse-LLM output.
    """

    def analyze_llm(self, outputfolder_path_list):
        """
        Analyze a single LLM output.
        Loads all data from the output folder, creates a summary with plots and saves it to the output folder.

        Args:
            outputfolder_path_list (list): List of output folder paths.

        """
        pass

    @staticmethod
    def load_metadata(metadata_path_list):
        """
        Load metadata from a CSV file into a DataFrame.

        Args:
            metadata_path_list (list): Path list with the metadata CSV files

        Returns:
            pd.DataFrame: DataFrame containing the metadata.
        """
        df_mdata = pd.DataFrame()
        for m_path in metadata_path_list:
            try:
                df = pd.read_csv(m_path)
                # Extract model name, task, dataset, and timestamp from the metadata path
                match = re.search(
                    r"\\([^\\]+)_([^_]+)_([^_]+)_(\d{8}_\d{6})_metadata\.csv$", m_path
                )
                if match:
                    model_name, task, dataset, timestamp = match.groups()
                    print(
                        f"Model Name: {model_name}, Task: {task}, Dataset: {dataset}, Timestamp: {timestamp}"
                    )
                    # Add extracted metadata to the DataFrame
                    df["model_name"] = model_name
                    df["task"] = task
                    df["dataset"] = dataset
                    df["timestamp"] = timestamp

                else:
                    print(
                        "Failed to extract metadata details from the path. Using default values."
                    )
                    df["model_name"] = "Unknown"
                    df["task"] = "Unknown"
                    df["dataset"] = "Unknown"
                    df["timestamp"] = "Unknown"

                # Append the DataFrame to the main DataFrame
                df_mdata = pd.concat([df_mdata, df], ignore_index=True)

            except Exception as e:
                print(f"Error loading metadata: {e}")
                continue
        return df_mdata

    @staticmethod
    def load_metrics_report_as_df(metrics_report_path):
        """
        Load metrics report from a JSON file and return it as a DataFrame.

        Args:
            metrics_report_path (str): Path to the metrics report JSON file.

        Returns:
            pd.DataFrame: DataFrame containing the metrics report.
        """
        try:
            with open(metrics_report_path, "r") as f:
                metrics_report = json.load(f)
            df = pd.DataFrame(metrics_report)
            # Expand the metrics_summary dict into separate columns
            if "metrics_summary" in df.columns:
                metrics_expanded = df["metrics_summary"].apply(
                    lambda x: x.get("overall", {}) if isinstance(x, dict) else {}
                )
                metrics_df = pd.json_normalize(metrics_expanded)
                # Prefix columns with "metric_"
                metrics_df = metrics_df.add_prefix("metric_")
                df = pd.concat([df, metrics_df], axis=1)
                # Drop the original metrics_summary column
                df.drop(columns=["metrics_summary"], inplace=True)
            return df
        except Exception as e:
            print(f"Error loading metrics report: {e}")
            return None

    @staticmethod
    def load_metrics_from_prompt_approaches(base_output_dir, prompt_approaches_paths):
        """
        Load metrics from different prompt approaches and concatenate them into a single DataFrame.
        Args:
            base_output_dir (str): Base directory containing the prompt approaches.
            prompt_approaches_paths (list): List of paths to the prompt approaches.

        Returns:
            pd.DataFrame: A DataFrame containing the concatenated metrics from all approaches.
        """
        df_results = pd.DataFrame()

        # Load metrics for each approach and concatenate them into a single DataFrame
        for approach in prompt_approaches_paths:
            approach_path = os.path.join(base_output_dir, approach)
            metrics_files = [
                f for f in os.listdir(approach_path) if "metrics_report.json" in f
            ]
            metrics_path = os.path.join(approach_path, metrics_files[0])
            df_temp = LLMAnalyzer.load_metrics_report_as_df(metrics_path)
            df_results = pd.concat([df_results, df_temp], ignore_index=True)

        # Plot the results

        return df_results

    @staticmethod
    def get_predictions(df, model=None, task=None, dataset=None, step_name=None):
        """
        Return filtered predictions DataFrame by model, task, and dataset if specified.

        Args:
            df (pd.DataFrame): DataFrame containing predictions and metadata columns.
            model (str, optional): Model name to filter by.
            task (str, optional): Task name to filter by.
            dataset (str, optional): Dataset name to filter by.
            step_name (str, optional): Step name to filter by for agents.

        Returns:
            pd.DataFrame: Filtered DataFrame according to specified parameters.
        """
        filtered_df = df
        if model is not None:
            filtered_df = filtered_df[filtered_df["model_name"] == model]
        if task is not None:
            filtered_df = filtered_df[filtered_df["task"] == task]
        if dataset is not None:
            filtered_df = filtered_df[filtered_df["dataset"] == dataset]
        if step_name is not None:
            filtered_df = filtered_df[filtered_df["step_name"] == step_name]
        return filtered_df

    @staticmethod
    def check_probability_inconsistency(df_mdata):
        """
        Check if the probability values match the classification labels in the metadata DataFrame.

        Args:
            df_mdata (pd.DataFrame): DataFrame containing metadata with 'Predicted Probability' and 'Predicted Diagnosis' columns.

        Returns:
            float: The proportion of predictions that are inconsistent with the probability.
        """
        counter = 0
        # Check for probability < 0.5 and "not" not in diagnosis
        mask_low = df_mdata["Predicted Probability"] < 0.5
        for diag in df_mdata.loc[mask_low, "Predicted Diagnosis"]:
            if "not" not in diag:
                counter += 1
        # Check for probability > 0.5 and "not" not in diagnosis
        mask_high = df_mdata["Predicted Probability"] > 0.5
        for diag in df_mdata.loc[mask_high, "Predicted Diagnosis"]:
            if "not" in diag:
                counter += 1

        print(
            f"{counter}/{len(df_mdata)} predictions are inconsistent with the probability."
        )

        return counter / len(df_mdata)

    @staticmethod
    def find_best_prompting_id(output_folder, model_list, metric="auprc"):
        """
        For each model in model_list, search for all files named 'metrics_report.json' in its subfolders,
        load them, and find the prompting_id with the best value for the given metric.

        Args:
            output_folder (str): Path to the parent folder containing model folders.
            model_list (list): List of model folder names.
            metric (str): Metric to use for selecting the best prompting_id (default: 'auprc').

        Returns:
            pd.DataFrame: {model_name: {'prompting_id': ..., 'metric_value': ..., 'metrics_report_path': ...}}
        """
        best_results = {}
        for model in model_list:
            model_path = os.path.join(output_folder, model)
            best_metric = -float("inf")
            best_prompting_id = None
            best_report_path = None

            # Walk through all subfolders to find metrics_report.json files
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if "metrics_report.json" in file:
                        report_path = os.path.join(root, file)
                        try:
                            with open(report_path, "r") as f:
                                reports = json.load(f)

                            # reports is a list of dicts, each with 'prompting_id' and metric keys
                            metric_values = []
                            prompting_id = None
                            for report in reports:
                                if prompting_id is None:
                                    prompting_id = report.get("prompting_id", None)
                                metrics = report["metrics_summary"]["overall"]
                                value = metrics.get(metric, None)
                                if value is not None:
                                    metric_values.append(value)
                            if metric_values:
                                mean_metric = np.mean(metric_values)
                                if mean_metric > best_metric:
                                    best_metric = mean_metric
                                    best_prompting_id = prompting_id
                                    best_report_path = report_path
                        except Exception as e:
                            print(f"Error reading {report_path}: {e}")

            if best_prompting_id is not None:
                best_results[model] = {
                    "prompting_id": best_prompting_id,
                    "metric_value": best_metric,
                    "metrics_report_path": best_report_path,
                }
            else:
                best_results[model] = None  # No valid report found

        df_best_results = pd.DataFrame.from_dict(best_results)

        return df_best_results

    @staticmethod
    def print_approach_summary(df, filters=None):
        """
        Print a summary of the approach from the metadata DataFrame.

        Parameters:
            df (pd.DataFrame): Metadata DataFrame (e.g., df_mdata).
            filters (dict): Optional. Dictionary of filters, e.g., {'task': ['aki', 'sepsis'], 'dataset': 'eicu'}.

        Prints:
            Number of samples, unique tasks, datasets, and a preview of predictions.
        """
        df_filtered = df.copy()
        if filters:
            for key, value in filters.items():
                if key in df_filtered.columns:
                    if isinstance(value, list):
                        df_filtered = df_filtered[df_filtered[key].isin(value)]
                    else:
                        df_filtered = df_filtered[df_filtered[key] == value]

        # Group by task and dataset
        group_cols = ["task", "dataset"]
        summary = (
            df_filtered.groupby(group_cols)
            .agg(
                mean_inference_time=("Inference Time", "mean"),
                total_inference_time=("Inference Time", "sum"),
                mean_input_tokens=("Input Tokens", "mean"),
                total_input_tokens=("Input Tokens", "sum"),
                mean_output_tokens=("Output Tokens", "mean"),
                total_output_tokens=("Output Tokens", "sum"),
                positive_samples=("Target Label", lambda x: (x == 1).sum()),
                negative_samples=("Target Label", lambda x: (x == 0).sum()),
                total_samples=("Target Label", "count"),
            )
            .reset_index()
        )

        summary_html = f"""
        <h3>Summary of Approach</h3>
        <ul>
            <li><b>Number of samples:</b> {len(df_filtered):,}</li>
            <li><b>Tasks:</b> {', '.join(map(str, df_filtered['task'].unique()))}</li>
            <li><b>Datasets:</b> {', '.join(map(str, df_filtered['dataset'].unique()))}</li>
            <li><b>Predicted diagnoses:</b> {', '.join(map(str, df_filtered['Predicted Diagnosis'].unique()))}</li>
            <li><b>Total inference time:</b> {df_filtered['Inference Time'].sum() / 3600:,.2f} h</li>
            <li><b>Total input tokens:</b> {df_filtered['Input Tokens'].sum():,}</li>
            <li><b>Total output tokens:</b> {df_filtered['Output Tokens'].sum():,}</li>
        </ul>
        """
        display(HTML(summary_html))

        # Calculate ratio of positive vs negative samples
        summary["pos_neg_ratio"] = summary["positive_samples"] / summary[
            "negative_samples"
        ].replace(0, np.nan)

        # Display summary as a styled table for better readability
        styled_summary = summary.style.format(
            {
                "mean_inference_time": "{:.2f}s",
                "total_inference_time": "{:.2f}s",
                "mean_input_tokens": "{:.0f}",
                "total_input_tokens": "{:.0f}",
                "mean_output_tokens": "{:.0f}",
                "total_output_tokens": "{:.0f}",
                "pos_neg_ratio": "{:.2f}",
            }
        ).background_gradient(
            subset=[
                "mean_inference_time",
                "mean_input_tokens",
                "mean_output_tokens",
                "pos_neg_ratio",
            ],
            cmap="Blues",
        )
        display(styled_summary)

    @staticmethod
    def plot_prediction_distribution(
        df,
        title_prefix="Prediction Probability Distribution",
        data_filter=None,
        bins=np.arange(0, 1.05, 0.05),
        color_neg="#4682b4",
        color_pos="salmon",
        show_stats=True,
        show_plot=True,
    ):
        """
        Plot the distribution of predicted probabilities, separated by ground truth labels.
        Ground truth positive and negative counts are displayed with striped histogram bars.

        Args:
            df (pd.DataFrame): DataFrame with columns 'Predicted Probability' and 'Target Label'.
            title (str): Main title for the plot.
            data_filter (dict): Optional. Dictionary of filters to apply to the DataFrame.
                e.g., {'task': 'aki', 'dataset': 'eicu'}
            bins (array): Bins for the histogram.
            color_neg (str): Color for true negative samples.
            color_pos (str): Color for true positive samples.
            show_stats (bool): If True, print summary statistics to stdout.
            show_plot (bool): If True, display the plot using plt.show().
        """
        df = df.copy()

        # Apply data_filter if provided
        if data_filter:
            for column, allowed_values in data_filter.items():
                if column in df.columns:
                    if isinstance(allowed_values, list):
                        df = df[df[column].isin(allowed_values)]
                    else:
                        df = df[df[column] == allowed_values]

        # Build a human-readable filter_config string
        filter_config = ""
        if data_filter:
            filter_parts = []
            for column, values in data_filter.items():
                display_key = VARIABLE_NAMES.get(str(column), str(column))

                if isinstance(values, list):
                    values_display = [
                        VARIABLE_NAMES.get(str(v), str(v)) for v in values
                    ]
                else:
                    values_display = [VARIABLE_NAMES.get(str(values), str(values))]

                filter_parts.append(", ".join(values_display))

            filter_config = " | ".join(filter_parts)

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )  # Slightly larger figure for better readability

        # Separate data by ground truth label
        positive_samples = df[df["Target Label"] == 1]
        negative_samples = df[df["Target Label"] == 0]

        # Stacked histogram for ground truth labels with hatching
        # True Negatives: Target Label == 0 and Predicted Probability < 0.5
        true_negative_samples = df[
            (df["Target Label"] == 0) & (df["Predicted Probability"] < 0.5)
        ]
        # True Positives: Target Label == 1 and Predicted Probability >= 0.5
        true_positive_samples = df[
            (df["Target Label"] == 1) & (df["Predicted Probability"] >= 0.5)
        ]
        # False Negatives: Target Label == 1 and Predicted Probability < 0.5
        false_negative_samples = df[
            (df["Target Label"] == 1) & (df["Predicted Probability"] < 0.5)
        ]
        # False Positives: Target Label == 0 and Predicted Probability >= 0.5
        false_positive_samples = df[
            (df["Target Label"] == 0) & (df["Predicted Probability"] >= 0.5)
        ]

        # Plot histograms for positive and negative samples
        ax.hist(
            [
                negative_samples["Target Label"],
                positive_samples["Target Label"],
            ],
            bins=bins,
            alpha=1.0,
            label=["Negative Samples (Label=0)", "Positive Samples (Label=1)"],
            color=[color_neg, color_pos],
            edgecolor="black",
            linewidth=1.0,
            stacked=True,
            hatch=["..", ".."],
        )

        # Plot histograms for each group with hatching for TP, TN, FP, FN
        ax.hist(
            [
                false_positive_samples["Predicted Probability"],
                true_positive_samples["Predicted Probability"],
            ],
            bins=bins,
            alpha=0.7,
            label=["False Positives (FP)", "True Positives (TP)"],
            color=[color_pos, color_pos],
            edgecolor=["black", "black"],
            linewidth=0.8,
            stacked=True,
            hatch=["xx", None],
        )

        # Overlay FP and FN with different hatching and lower alpha
        ax.hist(
            [
                false_negative_samples["Predicted Probability"],
                true_negative_samples["Predicted Probability"],
            ],
            bins=bins,
            alpha=0.7,
            label=["False Negatives (FN)", "True Negatives (TN)"],
            color=[color_neg, color_neg],
            edgecolor=["black", "black"],
            linewidth=0.8,
            stacked=True,
            hatch=["xx", None],
        )

        # Overlay all predicted probabilities as a step plot
        ax.hist(
            df["Predicted Probability"],
            bins=bins,
            alpha=0.7,
            # label="All Predicted Probabilities",
            edgecolor="black",
            color="black",
            histtype="step",
            linewidth=1.0,
        )

        # Add vertical line at 0.5 for decision threshold
        # ax.axvline(
        #     x=0.5,
        #     color="gray",
        #     linestyle="-",
        #     alpha=0.8,
        #     linewidth=1.5,
        #     label="Decision Threshold (0.5)",
        # )

        # Ground truth positive rate line
        ground_truth_positive_rate = df["Target Label"].mean()
        # ax.axvline(
        #     x=ground_truth_positive_rate,
        #     color="black",
        #     linestyle="dotted",
        #     alpha=0.8,
        #     linewidth=1.5,
        #     label=f"Ground Truth Positive Rate ({ground_truth_positive_rate:.3f})",
        # )

        # Mean predicted probability line
        mean_predicted_probability = df["Predicted Probability"].mean()
        # ax.axvline(
        #     x=mean_predicted_probability,
        #     color="black",
        #     linestyle="--",
        #     alpha=0.8,
        #     linewidth=1.5,
        #     label=f"Mean Predicted Probability ({mean_predicted_probability:.3f})",
        # )

        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{title_prefix}\n{filter_config}", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks(np.arange(0, 1.1, 0.1))

        # Print statistics
        if show_stats:
            pred_std = df["Predicted Probability"].std()
            calibration_error = abs(
                mean_predicted_probability - ground_truth_positive_rate
            )
            print(f"--- Prediction Distribution Statistics ---")
            print(f"Total Records: {len(df)}")
            print(f"Ground Truth Positive Samples: {len(positive_samples)}")
            print(f"Ground Truth Negative Samples: {len(negative_samples)}")
            print(
                f"Mean Predicted Probability (Average of All Predictions): {mean_predicted_probability:.3f}"
            )
            print(f"Std Predicted Probability: {pred_std:.3f}")
            print(
                f"Calibration Error (|Mean Pred. Prob. - Ground Truth Pos. Rate|): {calibration_error:.3f}"
            )
            print(f"True Positives: {len(true_positive_samples)}")
            print(f"True Negatives: {len(true_negative_samples)}")
            print(f"False Positives: {len(false_positive_samples)}")
            print(f"False Negatives: {len(false_negative_samples)}")

            # Calculate and print accuracy if possible (requires thresholding)
            y_pred_binary = (df["Predicted Probability"] >= 0.5).astype(int)
            accuracy = (y_pred_binary == df["Target Label"]).mean()
            print(f"Accuracy (at 0.5 threshold): {accuracy:.3f}")

            # This part assumes 'calculate_auprc' is defined or imported
            if "calculate_auprc" in globals():
                try:
                    auprc_val = calculate_auprc(
                        df["Target Label"].values, df["Predicted Probability"].values
                    )
                    if isinstance(auprc_val, dict):
                        auprc_value = auprc_val.get("auprc", 0.0)
                    else:
                        auprc_value = auprc_val
                    print(f"AUPRC: {auprc_value:.3f}")
                except Exception as e:
                    print(
                        f"Could not calculate AUPRC: {e}. Make sure sklearn.metrics.average_precision_score is available."
                    )

        if show_plot:
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_metrics(
        df,
        group=["model_id"],
        metrics=None,
        title_prefix="",
        figsize=(12, 8),
        save_path=None,
        dpi=300,
        style="seaborn-v0_8-whitegrid",
    ):
        """
        Plot metrics from the DataFrame, grouped by group with enhanced styling for presentations.

        Args:
            df (DataFrame): DataFrame containing metrics.
            group (list): List of columns to group by (default is ["model_id"]).
            metrics (list): Specific metrics to plot (optional). Defaults to AUPRC, AUROC, MCC.
            title_prefix (str): Prefix for the plot title.
            figsize (tuple): Figure size (width, height).
            save_path (str): Path to save the figure (optional).
            dpi (int): DPI for saved figure.
            style (str): Matplotlib style to use.
        """
        if df.empty:
            print("No data to plot.")
            return

        # Sort df by group
        # Custom sort order for 'task'
        if "task_id" in group:
            task_order = ["mortality", "aki", "sepsis"]
            df["task_id"] = pd.Categorical(
                df["task_id"], categories=task_order, ordered=True
            )
            df = df.sort_values(by=group)
        else:
            df = df.sort_values(by=group)

        # Set style for better presentation
        plt.style.use(style)

        # Ensure group is a list
        if isinstance(group, str):
            group = [group]

        # Determine which metrics to plot
        if metrics is not None:
            # Ensure metrics have the correct prefix
            metrics = [
                m if m.startswith("metric_") else f"metric_{m.lower()}" for m in metrics
            ]
            metric_labels = [m.replace("metric_", "").upper() for m in metrics]
        else:
            metrics = ["metric_auprc", "metric_auroc", "metric_mcc"]
            metric_labels = ["AUPRC", "AUROC", "MCC"]

        # Prepare data for plotting
        plot_data = []
        for idx, row in df.iterrows():
            for m, label in zip(metrics, metric_labels):
                if m in df.columns:
                    group_key = ", ".join(
                        VARIABLE_NAMES.get(str(row[g]), str(row[g])) for g in group
                    )
                    plot_data.append(
                        {
                            "Group": group_key,
                            "Metric": label,
                            "Value": row[m],
                        }
                    )

        plot_df = pd.DataFrame(plot_data)
        if plot_df.empty:
            print("No matching metrics found in DataFrame.")
            return

        # Sort by the order of appearance in the DataFrame for the group columns
        unique_groups = []
        seen = set()
        for key in plot_df["Group"]:
            if key not in seen:
                unique_groups.append(key)
                seen.add(key)
        plot_df["Group"] = pd.Categorical(
            plot_df["Group"], categories=unique_groups, ordered=True
        )

        # Aggregate duplicates if necessary
        agg_df = plot_df.groupby(["Group", "Metric"], as_index=False).agg(
            {"Value": ["mean", "count", "std"]}
        )
        # Flatten MultiIndex columns
        agg_df.columns = ["Group", "Metric", "Value", "Count", "Std"]
        agg_df["Std"] = agg_df["Std"].fillna(0)  # Fill NaN std with 0

        # Use only the mean values for plotting
        plot_df_final = agg_df[["Group", "Metric", "Value"]].pivot(
            index="Group", columns="Metric", values="Value"
        )

        # Get std values for error bars
        std_df = agg_df[["Group", "Metric", "Std"]].pivot(
            index="Group", columns="Metric", values="Std"
        )

        # Create figure with improved layout
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")

        # Create bar plot with error bars
        bars = plot_df_final.plot(
            kind="bar",
            ax=ax,
            width=0.7,
            edgecolor="white",
            linewidth=1.5,
            color=COLORS[: len(metric_labels)],
            yerr=std_df if agg_df["Count"].max() > 1 else None,
            capsize=4,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

        # Group labels
        group_names = [
            name[0] if isinstance(name, tuple) and len(name) == 1 else name
            for name in plot_df_final.index.tolist()
        ]

        # Use full names if not too many
        short_labels = [
            name[:20] + "..." if len(name) > 30 else name for name in group_names
        ]
        show_table = False

        ax.set_xticklabels(
            short_labels, rotation=45, ha="right", fontsize=11, fontweight="medium"
        )

        # Enhanced styling
        ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)

        # title and labels
        group_display_names = [
            VARIABLE_NAMES.get(g, g.replace("_", " ").title()) for g in group
        ]
        title = f"{title_prefix}Performance Metrics by {', '.join(group_display_names)}"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Groups", fontsize=12, fontweight="medium")
        ax.set_ylabel("Metric Value", fontsize=12, fontweight="medium")

        # Set y-axis limits and formatting
        ax.set_ylim(-0.2, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

        legend = ax.legend(
            metric_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=11,
            title="Metrics",
            title_fontsize=12,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)

        # Add value labels on bars - Fixed version
        # Get the bar containers (not ErrorbarContainer)
        bar_containers = [c for c in ax.containers if hasattr(c, "patches")]
        for container in bar_containers:
            # Get the actual bar heights for labeling
            labels = []
            for patch in container.patches:
                height = patch.get_height()
                if not np.isnan(height):
                    labels.append(f"{height:.3f}")
                else:
                    labels.append("")

            ax.bar_label(
                container,
                labels=labels,
                fontsize=9,
                rotation=90,
                padding=3,
                fontweight="medium",
            )

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=1.4)

        # Add subtle background
        ax.set_facecolor("#fafafa")

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        # Save figure if path provided
        if save_path:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            print(f"Figure saved to: {save_path}")

        plt.show()

    @staticmethod
    def save_metrics_to_results_json(
        metrics_report_path, results_json_path="../docs/results.json"
    ):
        """
        Save metrics from a metrics report file to the results.json file.

        If a model with the same model_id, prompting_id, and run_id already exists,
        it will be overwritten. Otherwise, new records are added to the results array.

        Args:
            metrics_report_path (str): Path to the metrics report JSON file
            results_json_path (str): Path to the results.json file (default: "../docs/results.json")

        Returns:
            tuple: (added_count, updated_count) - number of records added and updated
        """

        # Load existing results if file exists, otherwise start with empty dict
        if os.path.exists(results_json_path):
            with open(results_json_path, "r", encoding="utf-8") as f:
                try:
                    results_data = json.load(f)
                except json.JSONDecodeError:
                    results_data = {}
        else:
            results_data = {}

        # Ensure 'results' key exists and is a list
        if "results" not in results_data or not isinstance(
            results_data["results"], list
        ):
            results_data["results"] = []

        # Load metrics content from the metrics report file
        with open(metrics_report_path, "r", encoding="utf-8") as f:
            metrics_content = json.load(f)

        # Ensure metrics_content is a list
        if not isinstance(metrics_content, list):
            metrics_content = [metrics_content]

        added_count = 0
        updated_count = 0

        # Process each record in metrics_content
        for new_record in metrics_content:
            # Create a unique identifier for comparison
            new_id = (
                new_record.get("model_id", ""),
                # new_record.get("prompting_id", ""),
                new_record.get("run_id", ""),
                new_record.get("task_id", ""),
                new_record.get("dataset", ""),
            )

            # Look for existing record with same identifier
            existing_index = None
            for i, existing_record in enumerate(results_data["results"]):
                existing_id = (
                    new_record.get("model_id", ""),
                    # new_record.get("prompting_id", ""),
                    new_record.get("run_id", ""),
                    new_record.get("task_id", ""),
                    new_record.get("dataset", ""),
                )
                if existing_id == new_id:
                    existing_index = i
                    break

            # Update existing record or add new one
            if existing_index is not None:
                print(
                    f"Updating existing record: model_id='{new_record.get('model_id')}', "
                    f"prompting_id='{new_record.get('prompting_id')}', "
                    f"task_id='{new_record.get('task_id')}', "
                    f"dataset='{new_record.get('dataset')}', "
                    f"run_id='{new_record.get('run_id')}'"
                )
                results_data["results"][existing_index] = new_record
                updated_count += 1
            else:
                print(
                    f"Adding new record: model_id='{new_record.get('model_id')}', "
                    f"prompting_id='{new_record.get('prompting_id')}', "
                    f"task_id='{new_record.get('task_id')}', "
                    f"dataset='{new_record.get('dataset')}', "
                    f"run_id='{new_record.get('run_id')}'"
                )
                results_data["results"].append(new_record)
                added_count += 1

        # Save the updated results
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved to {results_json_path}")
        print(
            f"Added {added_count} new records, updated {updated_count} existing records"
        )
        print(f"Total records in results.json: {len(results_data['results'])}")

        return added_count, updated_count
