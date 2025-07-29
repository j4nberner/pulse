import logging
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.preprocessing.preprocessing_advanced.preprocessing_advanced import \
    PreprocessorAdvanced
from src.util.data_util import (get_all_feature_groups,
                                get_clinical_group_aliases,
                                get_common_feature_aliases,
                                get_feature_group_keys,
                                get_feature_group_title, get_feature_name,
                                get_feature_reference_range, get_feature_uom,
                                validate_feature_exists)

logger = logging.getLogger("PULSE_logger")

# ===========================
# FORMATTING FUNCTIONS
# ===========================


def format_clinical_data(
    patient_data: pd.Series,
    feature_keys: Set[str],
    preprocessor_advanced: PreprocessorAdvanced,
    include_demographics: bool = False,
    include_temporal_patterns: bool = False,
    include_uncertainty: bool = False,
    original_patient_data: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Format clinical data (vital signs or lab results) using aggregate_feature_windows.

    Args:
        patient_data: Patient data series (filtered, without _na columns)
        feature_keys: Set of feature keys to format
        preprocessor_advanced: Instance of PreprocessorAdvanced for data processing
        include_demographics: Whether to include demographics in output
        include_temporal_patterns: Whether to include temporal trend analysis
        include_uncertainty: Whether to include uncertainty/missingness analysis
        original_patient_data: Original patient data with _na columns for uncertainty analysis

    Returns:
        Dictionary with formatted clinical data
    """
    # Convert patient data to DataFrame for preprocessing
    patient_df = pd.DataFrame([patient_data])

    # Use aggregate_feature_windows to get min, max, mean for each feature
    aggregated_df = preprocessor_advanced.aggregate_feature_windows(patient_df)
    aggregated_row = aggregated_df.iloc[0]

    # Prepare result dictionary
    result = {}

    # Extract patient demographics if requested
    if include_demographics:
        patient_demographics = {}
        if "age" in patient_data.index:
            patient_demographics["age"] = patient_data["age"]
        if "sex" in patient_data.index:
            patient_demographics["sex"] = patient_data["sex"]
        if "height" in patient_data.index:
            patient_demographics["height"] = patient_data["height"]
        if "weight" in patient_data.index:
            patient_demographics["weight"] = patient_data["weight"]
        result["demographics"] = patient_demographics

    # Format clinical features with aggregated values using data_util
    clinical_data = {}

    for feature_key in feature_keys:
        # Check if this feature has aggregated columns in the data
        if any(col.startswith(f"{feature_key}_") for col in aggregated_row.index):
            min_val = aggregated_row.get(f"{feature_key}_min", None)
            max_val = aggregated_row.get(f"{feature_key}_max", None)
            mean_val = aggregated_row.get(f"{feature_key}_mean", None)

            # Only include features with non-NaN mean values
            if not pd.isna(mean_val):
                feature_name = get_feature_name(feature_key)
                unit = get_feature_uom(feature_key)
                normal_range = get_feature_reference_range(feature_key)

                clinical_data[feature_key] = {
                    "name": feature_name,
                    "min": min_val if not pd.isna(min_val) else mean_val,
                    "max": max_val if not pd.isna(max_val) else mean_val,
                    "mean": mean_val,
                    "unit": unit,
                    "normal_range": normal_range,
                }

                # Add temporal pattern analysis if requested
                if include_temporal_patterns:
                    temporal_pattern = analyze_temporal_pattern(
                        patient_data,
                        feature_key,
                        min_val,
                        max_val,
                        mean_val,
                        preprocessor_advanced,
                    )
                    clinical_data[feature_key]["temporal_pattern"] = temporal_pattern

                # Add uncertainty analysis if requested
                if include_uncertainty:
                    # Use original_patient_data (with _na columns) if available, otherwise fall back to patient_data
                    data_for_uncertainty = (
                        original_patient_data
                        if original_patient_data is not None
                        else patient_data
                    )
                    uncertainty_pattern = analyze_uncertainty_pattern(
                        data_for_uncertainty,
                        feature_key,
                    )
                    clinical_data[feature_key]["uncertainty"] = uncertainty_pattern

    # Store clinical data under appropriate key
    if include_demographics:
        result["vital_signs"] = clinical_data
    else:
        result = clinical_data

    return result


def format_clinical_text(clinical_data: Dict[str, Dict]) -> List[str]:
    """
    Format clinical data dictionary into human-readable text lines.

    Args:
        clinical_data: Dictionary with feature data containing name, min, max, mean, unit, normal_range

    Returns:
        List of formatted text lines
    """
    formatted_lines = []

    for feature_key, data in clinical_data.items():
        name = data["name"]
        min_val = data["min"]
        max_val = data["max"]
        mean_val = data["mean"]
        unit = data["unit"]
        normal_range = data["normal_range"]

        # Format value range or single value
        if abs(min_val - max_val) < 0.01:  # Essentially the same value
            value_str = f"{mean_val:.2f}"
        else:
            value_str = f"{min_val:.2f}-{max_val:.2f} (avg: {mean_val:.2f})"

        # Add normal range if available
        if unit and normal_range != (0, 0):
            normal_str = f" [normal: {normal_range[0]}-{normal_range[1]} {unit}]"
            base_text = f"- {name}: {value_str} {unit}{normal_str}"
        else:
            unit_str = f" {unit}" if unit else ""
            base_text = f"- {name}: {value_str}{unit_str}"

        # Add temporal pattern information if available
        pattern_info = []
        if "temporal_pattern" in data:
            pattern_info.append(data["temporal_pattern"])

        # Add uncertainty information if available
        if "uncertainty" in data:
            pattern_info.append(data["uncertainty"])

        if pattern_info:
            base_text += f" ({', '.join(pattern_info)})"

        formatted_lines.append(base_text)

    return formatted_lines


def format_demographics_str(demographics: dict) -> str:
    """
    Format demographics dictionary into a string for prompt templates.
    Always includes age, sex, height, and weight if present.
    Age, height, and weight are formatted as integers if possible.
    """
    demo_text = []
    if "age" in demographics and demographics["age"] is not None:
        age_val = int(round(float(demographics["age"])))
        demo_text.append(f"Age: {age_val} years")
    if "sex" in demographics and demographics["sex"] is not None:
        demo_text.append(f"Sex: {demographics['sex']}")
    if "height" in demographics and demographics["height"] is not None:
        height_val = int(round(float(demographics["height"])))
        demo_text.append(f"Height: {height_val} cm")
    if "weight" in demographics and demographics["weight"] is not None:
        weight_val = int(round(float(demographics["weight"])))
        demo_text.append(f"Weight: {weight_val} kg")
    return ", ".join(demo_text) if demo_text else "Demographics: Not available"


# ===========================
# TOOLS FUNCTIONS
# ===========================


def analyze_temporal_pattern(
    patient_data: pd.Series,
    feature_key: str,
    min_val: float,
    max_val: float,
    mean_val: float,
    preprocessor_advanced: PreprocessorAdvanced,
) -> str:
    """
    Simple temporal pattern analysis returning brief text assessment.

    Args:
        patient_data: Full patient data series
        feature_key: The feature to analyze (e.g., 'hr', 'sbp')
        min_val: Minimum value over monitoring period
        max_val: Maximum value over monitoring period
        mean_val: Mean value over monitoring period
        preprocessor_advanced: Instance of PreprocessorAdvanced for categorization

    Returns:
        Brief text description of trend and normality
    """
    try:
        # Extract time-windowed values for this feature
        time_series_values = []
        for col in patient_data.index:
            if col.startswith(f"{feature_key}_") and col.split("_")[-1].isdigit():
                val = patient_data[col]
                if not pd.isna(val):
                    time_series_values.append(val)

        if len(time_series_values) < 3:
            return "stable trend"

        # Linear regression for trend analysis
        n = len(time_series_values)
        x = np.arange(n)  # Time points
        y = np.array(time_series_values)

        # Calculate linear regression slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Normalize slope by mean value to get relative change rate
        relative_slope = (slope / y_mean * 100) if y_mean != 0 else 0

        # Determine trend direction and strength
        if abs(relative_slope) < 2:  # Less than 2% change per time unit
            trend = "stable"
        elif relative_slope >= 8:  # Strong increase (>=8% per time unit)
            trend = "rapidly increasing"
        elif relative_slope >= 4:  # Moderate increase (4-8% per time unit)
            trend = "moderately increasing"
        elif relative_slope > 0:  # Mild increase (2-4% per time unit)
            trend = "slowly increasing"
        elif relative_slope <= -8:  # Strong decrease (<=-8% per time unit)
            trend = "rapidly decreasing"
        elif relative_slope <= -4:  # Moderate decrease (-8 to -4% per time unit)
            trend = "moderately decreasing"
        else:  # Mild decrease (-4 to -2% per time unit)
            trend = "slowly decreasing"

        # Use categorize_features for abnormality assessment
        patient_df = pd.DataFrame([patient_data])
        categorized_df = preprocessor_advanced.categorize_features(
            df=patient_df,
            base_features={feature_key},
            X_cols=patient_data.index,
            num_categories=5,
            for_llm=True,
        )

        status = (
            categorized_df[feature_key].iloc[0]
            if feature_key in categorized_df.columns
            else "unknown range"
        )

        return f"values {status}, temporal trend {trend}"

    except Exception as e:
        logger.warning("Error analyzing temporal pattern for %s: %s", feature_key, e)
        return "stable trend"


def analyze_uncertainty_pattern(
    patient_data: pd.Series,
    feature_key: str,
) -> str:
    """
    Analyze missingness/imputation uncertainty for a feature using _na indicators.

    Args:
        patient_data: Full patient data series (with _na columns preserved)
        feature_key: The feature to analyze (e.g., 'hr', 'sbp')

    Returns:
        Brief text description of data completeness and uncertainty
    """
    try:
        # Look for windowed _na indicator columns (feature_na_0, feature_na_1, etc.)
        na_indicator_pattern = f"{feature_key}_na_"
        na_indicator_cols = [
            col
            for col in patient_data.index
            if col.startswith(na_indicator_pattern) and col.split("_")[-1].isdigit()
        ]

        # Debug: Check what _na columns are available
        if not na_indicator_cols:
            logger.debug("No windowed _na columns found for %s", feature_key)
            # No _na indicator found, check if base feature exists
            if any(col.startswith(f"{feature_key}_") for col in patient_data.index):
                return "complete data"
            else:
                return "feature not available"

        # Count total number of time windows for this feature
        time_window_cols = [
            col
            for col in patient_data.index
            if col.startswith(f"{feature_key}_")
            and col.split("_")[-1].isdigit()
            and not "_na_" in col
        ]
        total_windows = len(time_window_cols)

        if total_windows == 0:
            return "feature not available"

        # Calculate how many windows were imputed by checking _na indicators
        imputed_windows = 0
        for na_col in na_indicator_cols:
            na_value = patient_data[na_col]
            if not pd.isna(na_value):
                try:
                    # If _na indicator is 1, this window was imputed
                    if float(na_value) == 1.0:
                        imputed_windows += 1
                except (ValueError, TypeError):
                    continue

        # Create detailed uncertainty description
        if imputed_windows == 0:
            completeness = "complete data"
        elif imputed_windows == total_windows:
            completeness = f"fully imputed ({imputed_windows}/{total_windows})"
        else:
            completeness = f"partially imputed ({imputed_windows}/{total_windows})"

        return completeness

    except Exception as e:
        logger.warning("Error analyzing uncertainty for %s: %s", feature_key, e)
        return "unknown completeness"


# ===========================
# DATA COLLECTION UTILS
# ===========================


def get_available_vitals(available_features: Set[str]) -> Set[str]:
    """Get available vital signs from the data using data_util groups."""
    available_vitals = set()
    vitals_keys = get_feature_group_keys("vitals")

    for vital in vitals_keys:
        if any(feat.startswith(vital) for feat in available_features):
            available_vitals.add(vital)
    return available_vitals


def get_available_labs(
    requested_labs: List[str], available_features: Set[str]
) -> Set[str]:
    """Get available requested labs from the data."""
    available_labs = set()
    for lab in requested_labs:
        if any(feat.startswith(lab) for feat in available_features):
            available_labs.add(lab)
    return available_labs


def get_lab_groups_available(available_features: Set[str]) -> Dict[str, List[str]]:
    """Get available lab tests organized by clinical groups."""
    available_by_group = {}
    lab_groups = get_all_feature_groups()

    for group_name, group_dict in lab_groups.items():
        if group_name == "vitals":  # Skip vitals as they're handled separately
            continue

        available_in_group = []
        for feature_key in group_dict.keys():
            if any(feat.startswith(feature_key) for feat in available_features):
                available_in_group.append(feature_key)

        if available_in_group:
            available_by_group[group_name] = available_in_group

    return available_by_group


def validate_features(feature_list: List[str]) -> List[str]:
    """Validate that requested features exist in the data_util feature dictionary."""
    valid_features = []
    lab_groups = get_all_feature_groups()

    # Get mappings from data_util
    group_mappings = {}

    # Add official group mappings from data_util
    for group_key in [
        "bga",
        "coag",
        "electrolytes_met",
        "liver_kidney",
        "hematology_immune",
        "cardiac",
    ]:
        group_features = get_feature_group_keys(group_key)
        group_title = get_feature_group_title(group_key).lower()

        # Add both the key and the display title as mappings
        group_mappings[group_key] = group_features
        group_mappings[group_title] = group_features

    # Add clinical aliases from data_util
    clinical_aliases = get_clinical_group_aliases()
    for feature_tuple, aliases in clinical_aliases.items():
        feature_list_from_tuple = list(feature_tuple)
        for alias in aliases:
            group_mappings[alias] = feature_list_from_tuple

    # Get individual feature name mappings from data_util
    name_mappings = get_common_feature_aliases()

    for feature in feature_list:
        feature_lower = feature.lower().strip()

        # First check exact match (highest priority)
        if validate_feature_exists(feature):
            valid_features.append(feature)
            continue

        # Check lowercase exact match
        if validate_feature_exists(feature_lower):
            valid_features.append(feature_lower)
            continue

        # Check if it's a group name that needs expansion
        if feature_lower in group_mappings:
            group_features = group_mappings[feature_lower]
            added_features = []
            for group_feature in group_features:
                if validate_feature_exists(group_feature):
                    valid_features.append(group_feature)
                    added_features.append(group_feature)
            if added_features:
                logger.info(
                    "Expanded group '%s' to features: %s", feature, added_features
                )
            continue

        # Check if it needs individual mapping
        if feature_lower in name_mappings:
            mapped_feature = name_mappings[feature_lower]
            if validate_feature_exists(mapped_feature):
                valid_features.append(mapped_feature)
                logger.info("Mapped '%s' to '%s'", feature, mapped_feature)
                continue
            else:
                logger.warning(
                    "Mapped feature '%s' not found in data_util", mapped_feature
                )
                continue

        # Check for partial matches in available feature groups
        found_match = False
        for group_name, group_dict in lab_groups.items():
            for feature_key in group_dict.keys():
                if (
                    feature_lower in feature_key.lower()
                    or feature_key.lower() in feature_lower
                ):
                    if validate_feature_exists(feature_key):
                        valid_features.append(feature_key)
                        logger.info(
                            "Partial match: mapped '%s' to '%s'",
                            feature,
                            feature_key,
                        )
                        found_match = True
                        break
            if found_match:
                break

        if not found_match:
            logger.warning("Feature '%s' not found in data_util", feature)

    return valid_features


def validate_lab_request(requested_labs: List[str], state: Dict[str, Any]) -> List[str]:
    """
    Validate lab requests to ensure no already-used features are requested again.

    Args:
        requested_labs: List of requested lab abbreviations
        state: Current workflow state

    Returns:
        List of valid, unused lab abbreviations
    """
    valid_labs = []
    already_used_requests = []

    for lab in requested_labs:
        if lab in state["used_features"]:
            already_used_requests.append(lab)
        else:
            valid_labs.append(lab)

    if already_used_requests:
        logger.warning(
            "Model requested already analyzed features: %s. These will be ignored.",
            already_used_requests,
        )
        logger.info("Valid new lab requests: %s", valid_labs)

    return valid_labs


def extract_requested_labs(lab_order_output: Dict[str, Any]) -> Optional[List[str]]:
    """Extract requested labs from lab ordering output with validation."""
    if not isinstance(lab_order_output, dict):
        logger.info("Lab ordering failed to return dict, stopping iteration")
        return None

    if "requested_tests" in lab_order_output:
        return lab_order_output.get("requested_tests", [])
    elif lab_order_output.get("diagnosis") == "unknown":
        # This is the fallback dict from failed parsing
        logger.info("Lab ordering failed to parse JSON, stopping iteration")
        return None
    else:
        # Unexpected dict format
        logger.info("Lab ordering returned unexpected format, stopping iteration")
        return None


# ===========================
# SPECIALIST AGENT UTILS
# ===========================


def get_specialist_features(
    specialist_type: str, available_features: Set[str]
) -> Set[str]:
    """Get available features for a specific specialist agent type."""
    specialist_config = {
        "hemodynamic": ["vitals", "cardiac"],
        "metabolic": ["bga", "electrolytes_met", "liver_kidney"],
        "hematologic": ["hematology_immune", "coag"],
    }

    if specialist_type not in specialist_config:
        return set()

    specialist_features = set()
    for group_name in specialist_config[specialist_type]:
        group_keys = get_feature_group_keys(group_name)
        for feature_key in group_keys:
            if any(feat.startswith(feature_key) for feat in available_features):
                specialist_features.add(feature_key)

    return specialist_features


def get_specialist_system_message(specialist_type: str, task_name: str) -> str:
    """Get system message for specialist agents."""

    specialist_descriptions = {
        "hemodynamic": "You are a hemodynamic specialist analyzing cardiovascular stability, perfusion status, and cardiac function to assess {task_name} risk. Focus on vital signs patterns, cardiac markers, and circulatory indicators. Provide concise, focused assessments.",
        "metabolic": "You are a metabolic specialist analyzing acid-base balance, electrolyte status, and organ function to assess {task_name} risk. Focus on blood gas analysis, metabolic markers, and organ dysfunction indicators. Provide concise, focused assessments.",
        "hematologic": "You are a hematology specialist analyzing blood counts, immune response, and coagulation status to assess {task_name} risk. Focus on hematologic parameters, inflammatory markers, and bleeding risk indicators. Provide concise, focused assessments.",
    }

    return specialist_descriptions.get(
        specialist_type,
        f"You are a {specialist_type} specialist analyzing clinical data to assess {task_name} risk.",
    ).format(task_name=task_name)


# ===========================
# TASK-SPECIFIC CONTENT
# ===========================


def get_task_specific_content(task_name: str) -> Dict[str, str]:
    """
    Get task-specific content for prompts.

    Args:
        task_name: The task name (e.g., 'mortality', 'aki', 'sepsis')

    Returns:
        Dictionary containing complication_name and task_info for the specified task
    """
    if task_name == "mortality":
        return {
            "task_name": "mortality",
            "complication_name": "death",
            "prediction_description": "the prediction of ICU mortality",
            "task_info": "ICU mortality refers to death occurring during the ICU stay. Key risk factors include hemodynamic instability, respiratory failure, multi-organ dysfunction, and severe metabolic derangements.",
            "task_info_long": "Mortality refers to the occurrence of death within a specific population and time period. In the context of ICU patients, the task involves analyzing information from the first 25 hours of a patient’s ICU stay to predict whether the patient will survive the remainder of their stay. This prediction task supports early risk assessment and clinical decision-making in critical care settings.",
        }
    elif task_name == "aki":
        return {
            "task_name": "aki",
            "complication_name": "acute kidney injury",
            "prediction_description": "prediction of the onset of acute kidney injury",
            "task_info": "Acute kidney injury (AKI) is defined by rapid decline in kidney function with increased creatinine (≥1.5x baseline or ≥0.3 mg/dL increase in 48h) or decreased urine output (<0.5 mL/kg/h for 6-12h). Common causes include sepsis, hypotension, and nephrotoxins.",
            "task_info_long": "Acute kidney injury (AKI) is a subset of acute kidney diseases and disorders (AKD), characterized by a rapid decline in kidney function occurring within 7 days, with health implications. According to KDIGO criteria, AKI is diagnosed when there is an increase in serum creatinine to ≥1.5 times baseline within the prior 7 days, or an increase in serum creatinine by ≥0.3 mg/dL (≥26.5 µmol/L) within 48 hours, or urine output <0.5 mL/kg/h for 6–12 hours. The most common causes of AKI include sepsis, ischemia from hypotension or shock, and nephrotoxic exposures such as certain medications or contrast agents.",
        }
    elif task_name == "sepsis":
        return {
            "task_name": "sepsis",
            "complication_name": "sepsis",
            "prediction_description": "prediction of the onset of sepsis",
            "task_info": "Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. It is diagnosed by an increase in the SOFA score of ≥2 points in the presence of suspected infection. Key indicators include fever, tachycardia, tachypnea, altered mental status, and laboratory abnormalities.",
            "task_info_long": "Sepsis is a life-threatening condition characterized by organ dysfunction resulting from a dysregulated host response to infection. It is diagnosed when a suspected or confirmed infection is accompanied by an acute increase of two or more points in the patient’s Sequential Organ Failure Assessment (SOFA) score relative to their baseline. The SOFA score evaluates six physiological parameters: the ratio of partial pressure of oxygen to the fraction of inspired oxygen, mean arterial pressure, serum bilirubin concentration, platelet count, serum creatinine level, and the Glasgow Coma Score. A complication of sepsis is septic shock, which is marked by a drop in blood pressure and elevated lactate levels. Indicators of suspected infection may include positive blood cultures or the initiation of antibiotic therapy.",
        }
    return {
        "complication_name": "complications",
        "task_info": "General ICU complications assessment.",
    }


# ===========================
# OTHER UTILS
# ===========================


def extract_confidence(output: Dict[str, Any]) -> float:
    """Extract confidence value from LLM output with fallback logic."""
    if "confidence" in output:
        confidence = output.get("confidence", 0)
        # Handle string values from LLM output
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 0
        return confidence / 100.0
    else:
        # Use probability as confidence indicator when confidence not provided
        probability = output.get("probability", 50)
        # Handle string values from LLM output
        if isinstance(probability, str):
            try:
                probability = float(probability)
            except ValueError:
                probability = 50
        return probability / 100.0


def create_error_response(error_message: str) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "generated_text": {"error": error_message},
        "token_time": 0,
        "infer_time": 0,
        "num_input_tokens": 0,
        "num_output_tokens": 0,
    }


def get_monitoring_period_hours(patient_data: pd.Series) -> int:
    """
    Get the monitoring period duration in hours from windowed data columns.

    Args:
        patient_data: Patient data series with windowed columns (e.g., feature_0, feature_1, etc.)

    Returns:
        Number of hours in the monitoring period (e.g., 6 if columns go from feature_0 to feature_5)
    """
    window_indices = set()
    for col in patient_data.index:
        parts = col.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            window_indices.add(int(parts[-1]))
    return len(window_indices)


def filter_na_columns(patient_data: pd.Series) -> pd.Series:
    """Filter out columns with '_na' suffixes like Sarvari preprocessor does."""
    # Convert to DataFrame for regex filtering, then back to Series
    temp_df = pd.DataFrame([patient_data])
    filtered_df = temp_df.filter(regex=r"^(?!.*_na(_\d+)?$)")
    return filtered_df.iloc[0]


def parse_numeric_value(value, default=0):
    """
    Parse a numeric value: return float if float, int if int, float from string, else default.
    If value is None or missing, return "unknown".
    """
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().strip("\"'")

        numeric_match = re.search(r"(-?\d+\.?\d*)", s)
        if numeric_match:
            try:
                num_str = numeric_match.group(1)
                if "." in num_str:
                    return float(num_str)
                else:
                    return int(num_str)
            except Exception:
                pass
    logger.warning(
        "Failed to parse numeric value from '%s', returning default: %s", value, default
    )
    return default
