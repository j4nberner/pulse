import glob
import json
import os
import re

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.table import Table
from matplotlib import gridspec

from src.eval.metrics import calculate_auprc


class LLMAnalyzer:
    """
    A class to analyze the performance of a Pulse-LLM output.
    """

    def __init__(self):
        """
        Initializes the LLM_Analyzer class.
        """
        pass

    def analyze_llm(self, outputfolder_path_list):
        """
        Analyze a single LLM output.
        Loads all data from the output folder, creates a summary with plots and saves it to the output folder.

        Args:
            outputfolder_path_list (list): List of output folder paths.

        """
        categorized_files = self.categorize_files(outputfolder_path_list)
        metadata_path_list = categorized_files["metadata_files"]
        metrics_report_path = categorized_files["metrics_report_files"]

        df_mdata = self.load_metadata(metadata_path_list)

    @staticmethod
    def categorize_files(outputfolder_path_list):
        """
        Categorize files in the output folders into metrics report files, metadata files, and log files.

        Args:
            outputfolder_path_list (list): List of output folder paths.

        Returns:
            dict: A dictionary containing categorized files.
        """
        file_list = []
        for outputfolder_path in outputfolder_path_list:
            file_list.extend(glob.glob(os.path.join(outputfolder_path, "*")))

        categorized_files = {
            "metrics_report_files": [f for f in file_list if "metrics_report" in f],
            "metadata_files": [f for f in file_list if "metadata" in f],
            "log_files": [f for f in file_list if "log" in f],
        }

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

    @staticmethod
    def load_metadata(metadata_path_list):
        """
        Load metadata from a CSV file into a DataFrame.

        Args:
            metadata_path (str): Path to the metadata CSV file.

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
    def get_predictions(df, model=None, task=None, dataset=None):
        """
        Return filtered predictions DataFrame by model, task, and dataset if specified.

        Args:
            df (pd.DataFrame): DataFrame containing predictions and metadata columns.
            model (str, optional): Model name to filter by.
            task (str, optional): Task name to filter by.
            dataset (str, optional): Dataset name to filter by.

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
    def plot_prediction_distribution(
        df,
        title="Prediction Distribution",
        bins=np.arange(0, 1.05, 0.05),
        color_neg="#4682b4",
        color_pos="salmon",
        show_stats=True,
        show=True,
    ):
        """
        Plot the distribution of predicted probabilities, separated by target label.

        Args:
            df (pd.DataFrame): DataFrame with columns 'Predicted Probability' and 'Target Label'.
            title (str): Plot title.
            bins (array): Histogram bins.
            color_neg (str): Color for negative samples.
            color_pos (str): Color for positive samples.
            show_stats (bool): Print statistics to stdout.
            show (bool): Whether to call plt.show().
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        # Separate by label
        positive_samples = df[df["Target Label"] == 1]
        negative_samples = df[df["Target Label"] == 0]

        # Stacked histogram for actual labels
        ax.hist(
            [
                negative_samples["Predicted Probability"],
                positive_samples["Predicted Probability"],
            ],
            bins=bins,
            alpha=0.7,
            label=["True Negative (Label=0)", "True Positive (Label=1)"],
            color=[color_neg, color_pos],
            edgecolor="black",
            linewidth=0.5,
            stacked=True,
        )

        # Overlay all predicted probabilities as step
        ax.hist(
            df["Predicted Probability"],
            bins=bins,
            alpha=0.7,
            label="All Predicted Probabilities",
            edgecolor="black",
            color="black",
            histtype="step",
            linewidth=1.0,
        )

        # Add vertical line at 0.5 for decision threshold
        ax.axvline(
            x=0.5,
            color="gray",
            linestyle="-",
            alpha=0.8,
            linewidth=1.5,
            label="Decision Threshold (0.5)",
        )

        # Positive rate line
        pos_rate = df["Target Label"].mean()
        ax.axvline(
            x=pos_rate,
            color="red",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            label=f"Positive Rate ({pos_rate:.3f})",
        )

        # Mean predicted probability
        pred_mean = df["Predicted Probability"].mean()
        ax.axvline(
            x=pred_mean,
            color="blue",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            label=f"Mean Pred. Prob. ({pred_mean:.3f})",
        )

        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{title}\n(n={len(df)})", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.2))

        # Print statistics
        if show_stats:
            pred_std = df["Predicted Probability"].std()
            calibration_error = abs(pred_mean - pos_rate)
            print(f"Records: {len(df)}")
            print(f"Positive samples: {len(positive_samples)}")
            print(f"Negative samples: {len(negative_samples)}")
            print(f"Positive rate (actual): {pos_rate:.3f}")
            print(f"Mean predicted probability: {pred_mean:.3f}")
            print(f"Std predicted probability: {pred_std:.3f}")
            print(
                f"Calibration error (|pred_mean - pos_rate|): {calibration_error:.3f}"
            )
            if "calculate_auprc" in globals():
                auprc_val = calculate_auprc(
                    df["Target Label"].values, df["Predicted Probability"].values
                )
                if isinstance(auprc_val, dict):
                    auprc_value = auprc_val.get("auprc", 0.0)
                else:
                    auprc_value = auprc_val
                print(f"AUPRC: {auprc_value:.3f}")

        if show:
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_metrics(df, group=["model_id"], metrics=None, titel_prefix=""):
        """
        Plot metrics from the DataFrame, grouped by group.

        Args:
            df (DataFrame): DataFrame containing metrics.
            group (list): List of columns to group by (default is ["model_id"]).
            metrics (list): Specific metrics to plot (optional). Defaults to AUPRC, AUROC, MCC.
            titel_prefix (str): Prefix for the plot title.
        """
        if df.empty:
            print("No data to plot.")
            return

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
                    group_key = tuple(row[g] for g in group)
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
        # and keep the order of unique group keys as they appear
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
            {"Value": ["mean", "count"]}
        )
        # Flatten MultiIndex columns
        agg_df.columns = ["Group", "Metric", "Value", "Count"]

        # Use only the mean values for plotting
        plot_df_final = agg_df[["Group", "Metric", "Value"]].pivot(
            index="Group", columns="Metric", values="Value"
        )

        # Determine the number of groups (rows in the table)
        n_groups = len(plot_df_final)
        # Adjust figure height: base height + extra per group (tune as needed)
        base_height = 6
        extra_height_per_group = 0.5
        fig_height = base_height + n_groups * extra_height_per_group

        ax = plot_df_final.plot(
            kind="bar",
            figsize=(10, fig_height),
            width=0.5,
            edgecolor="black",
            colormap="Set1",
        )
        # Shorten group labels using a mapping to single letters
        group_names = [
            name[0] if isinstance(name, tuple) and len(name) == 1 else name
            for name in plot_df_final.index.tolist()
        ]
        group_label_dict = {name: chr(65 + i) for i, name in enumerate(group_names)}
        short_labels = [group_label_dict[name] for name in group_names]
        ax.set_xticklabels(short_labels, rotation=0)

        # Print the mapping as a table
        # print("Group label mapping:")
        # print(pd.DataFrame({"Letter": short_labels, "Group": group_names}))

        # Add the mapping as a table below the plot

        # Prepare table data
        table_data = list(zip(short_labels, group_names))
        col_labels = ["Letter", "Group"]

        # Create a new axes below the main plot for the table

        # Get the current figure
        fig = plt.gcf()

        # Adjust the main plot to make space for the table
        plt.subplots_adjust(bottom=0.3)

        # Add the table below the plot using bbox_to_anchor
        table = plt.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="bottom",
            bbox=[0, -0.35, 1, 0.25],  # [left, bottom, width, height]
            colColours=["#f2f2f2"] * 2,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.title(f"{titel_prefix}Metrics Grouped by {', '.join(group)}")
        plt.xlabel("Group")
        plt.ylabel("Value")
        plt.ylim(0, 1)
        plt.legend(metric_labels, frameon=False)
        # plt.tight_layout()
        plt.show()
