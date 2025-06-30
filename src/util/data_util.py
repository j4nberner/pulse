# Dictionary of feature limits for outlier detection
from typing import Tuple

features_dict = {
    "hr": ("Heart Rate", "bpm", (60, 100)),
    "sbp": ("Systolic Blood Pressure", "mmHg", (90, 120)),
    "dbp": ("Diastolic Blood Pressure", "mmHg", (60, 80)),
    "map": ("Mean Arterial Pressure (MAP)", "mmHg", (65, 100)),
    "o2sat": ("Oxygen Saturation", "%", (95, 100)),
    "resp": ("Respiratory Rate", "breaths/min", (12, 20)),
    "temp": ("Temperature", "°C", (36.5, 37.5)),
    "ph": ("pH Level", "/", (7.35, 7.45)),
    "po2": ("Partial Pressure of Oxygen (PaO2)", "mmHg", (75, 100)),
    "pco2": ("Partial Pressure of Carbon Dioxide (PaCO2)", "mmHg", (35, 45)),
    "be": ("Base Excess", "mmol/L", (-2, 2)),
    "bicar": ("Bicarbonate", "mmol/L", (22, 29)),
    "fio2": ("Fraction of Inspired Oxygen (FiO2)", "%", (21, 100)),
    "inr_pt": ("International Normalized Ratio (INR)", "/", (0.8, 1.2)),
    "ptt": ("Partial Thromboplastin Time (PTT)", "sec", (25, 35)),
    "fgn": ("Fibrinogen", "mg/dL", (200, 400)),
    "na": ("Sodium", "mmol/L", (135, 145)),
    "k": ("Potassium", "mmol/L", (3.5, 5)),
    "cl": ("Chloride", "mmol/L", (96, 106)),
    "ca": ("Calcium", "mg/dL", (8.5, 10.5)),
    "cai": ("Ionized Calcium", "mmol/L", (1.1, 1.3)),
    "mg": ("Magnesium", "mg/dL", (1.7, 2.2)),
    "phos": ("Phosphate", "mg/dL", (2.5, 4.5)),
    "glu": ("Glucose", "mg/dL", (70, 140)),
    "lact": ("Lactate", "mmol/L", (0.5, 2)),
    "alb": ("Albumin", "g/dL", (3.5, 5)),
    "alp": ("Alkaline Phosphatase", "U/L", (44, 147)),
    "alt": ("Alanine Aminotransferase (ALT)", "U/L", (7, 56)),
    "ast": ("Aspartate Aminotransferase (AST)", "U/L", (10, 40)),
    "bili": ("Total Bilirubin", "mg/dL", (0.1, 1.2)),
    "bili_dir": ("Direct Bilirubin", "mg/dL", (0, 0.3)),
    "bun": ("Blood Urea Nitrogen (BUN)", "mg/dL", (7, 20)),
    "crea": ("Creatinine", "mg/dL", (0.6, 1.3)),
    "urine": ("Urine Output", "mL/h", (30, 50)),
    "hgb": ("Hemoglobin", "g/dL", (12.5, 17.5)),
    "mch": ("Mean Corpuscular Hemoglobin (MCH)", "pg", (27, 33)),
    "mchc": ("Mean Corpuscular Hemoglobin Concentration (MCHC)", "g/dL", (32, 36)),
    "mcv": ("Mean Corpuscular Volume (MCV)", "fL", (80, 100)),
    "plt": ("Platelets", "1000/µL", (150, 450)),
    "wbc": ("White Blood Cell Count (WBC)", "1000/µL", (4, 11)),
    "neut": ("Neutrophils", "%", (55, 70)),
    "bnd": ("Band Neutrophils", "%", (0, 6)),
    "lymph": ("Lymphocytes", "%", (20, 40)),
    "crp": ("C-Reactive Protein (CRP)", "mg/L", (0, 10)),
    "methb": ("Methemoglobin", "%", (0, 2)),
    "ck": ("Creatine Kinase (CK)", "U/L", (30, 200)),
    "ckmb": ("Creatine Kinase-MB (CK-MB)", "ng/mL", (0, 5)),
    "tnt": ("Troponin T", "ng/mL", (0, 14)),
    "height": ("Height", "cm", ()),
    "weight": ("Weight", "kg", ()),
}

# Feature groupings for dynamic features (excluding height and weight)
features_dyn_vitals_list = [
    "hr",
    "sbp",
    "dbp",
    "map",
    "o2sat",
    "resp",
    "temp",
]  # Vital Signs
features_dyn_bga_list = [
    "ph",
    "po2",
    "pco2",
    "be",
    "bicar",
    "fio2",
]  # Blood Gas Analysis
features_dyn_coag_list = ["inr_pt", "ptt", "fgn"]  # Coagulation
features_dyn_electrolytes_met_list = [  # Metabolic Panel & Electrolytes
    "na",
    "k",
    "cl",
    "ca",
    "cai",
    "mg",
    "phos",
    "glu",
    "lact",
]
features_dyn_liver_kidney_list = [  # Liver & Kidney Function
    "alb",
    "alp",
    "alt",
    "ast",
    "bili",
    "bili_dir",
    "bun",
    "crea",
    "urine",
]
features_dyn_hematology_immune_list = [  # Hematology & Immune Response
    "hgb",
    "mch",
    "mchc",
    "mcv",
    "plt",
    "wbc",
    "neut",
    "bnd",
    "lymph",
    "crp",
    "methb",
]
features_dyn_cardiac_list = ["ck", "ckmb", "tnt"]  # Cardiac Markers

# Group titles
features_dyn_sets_titles_dict = {
    "vitals": "Vital Signs",
    "bga": "Blood Gas Analysis",
    "coag": "Coagulation",
    "electrolytes_met": "Metabolic Panel & Electrolytes",
    "liver_kidney": "Liver & Kidney Function",
    "hematology_immune": "Hematology & Immune Response",
    "cardiac": "Cardiac Markers",
}

# Dynamic features dictionary (excluding static features like height and weight)
features_dyn_dict = {
    k: v for k, v in features_dict.items() if k not in ["height", "weight"]
}

# Create feature set dictionaries dynamically using the lists and the main dictionary
features_dyn_sets_dict = {
    "vitals": {f: features_dyn_dict[f] for f in features_dyn_vitals_list},
    "bga": {f: features_dyn_dict[f] for f in features_dyn_bga_list},
    "coag": {f: features_dyn_dict[f] for f in features_dyn_coag_list},
    "electrolytes_met": {
        f: features_dyn_dict[f] for f in features_dyn_electrolytes_met_list
    },
    "liver_kidney": {f: features_dyn_dict[f] for f in features_dyn_liver_kidney_list},
    "hematology_immune": {
        f: features_dyn_dict[f] for f in features_dyn_hematology_immune_list
    },
    "cardiac": {f: features_dyn_dict[f] for f in features_dyn_cardiac_list},
}

# ----------------------------------------
# Functions to access feature data
# ----------------------------------------


def _get_feature_info(feature_name: str) -> Tuple[str, str, Tuple[float, float]]:
    """Helper function to get feature information with proper fallback."""
    return features_dict.get(feature_name, (feature_name, "", (0, 0)))


def get_feature(feature_name: str) -> tuple:
    """Returns the converted feature for a given feature key."""
    return _get_feature_info(feature_name)


def get_feature_name(feature_name: str) -> str:
    """Returns the full feature name for a given feature key."""
    return _get_feature_info(feature_name)[0]


def get_feature_uom(feature_name: str) -> str:
    """Returns the feature unit of measurement (uom) for a given feature key."""
    return _get_feature_info(feature_name)[1]


def get_feature_reference_range(feature_name: str) -> Tuple[float, float]:
    """Returns the feature reference range for a given feature key."""
    return _get_feature_info(feature_name)[2]


def get_feature_group(group_name: str) -> dict:
    """Returns the feature dictionary for a specific group."""
    return features_dyn_sets_dict.get(group_name, {})


def get_feature_group_keys(group_name: str) -> list:
    """Returns the list of feature keys for a specific group."""
    group_lists = {
        "vitals": features_dyn_vitals_list,
        "bga": features_dyn_bga_list,
        "coag": features_dyn_coag_list,
        "electrolytes_met": features_dyn_electrolytes_met_list,
        "liver_kidney": features_dyn_liver_kidney_list,
        "hematology_immune": features_dyn_hematology_immune_list,
        "cardiac": features_dyn_cardiac_list,
    }
    return group_lists.get(group_name, [])


def get_all_feature_groups() -> dict:
    """Returns all feature groups."""
    return features_dyn_sets_dict


def get_feature_group_title(group_name: str) -> str:
    """Returns the display title for a specific group."""
    return features_dyn_sets_titles_dict.get(group_name, group_name)


# ----------------------------------------
# Feature validation and mapping utilities
# ----------------------------------------


def get_all_group_names() -> list:
    """Returns all available group names (both keys and display titles)."""
    group_names = list(features_dyn_sets_titles_dict.keys())
    group_titles = [title.lower() for title in features_dyn_sets_titles_dict.values()]
    return group_names + group_titles


def validate_feature_exists(feature_key: str) -> bool:
    """Check if a feature key exists in the features dictionary."""
    return feature_key in features_dict


def get_common_feature_aliases() -> dict:
    """Get common aliases for feature names that map to data_util keys."""
    return {
        # Full medical names to abbreviations
        "sodium": "na",
        "potassium": "k",
        "chloride": "cl",
        "calcium": "ca",
        "ionized calcium": "cai",
        "magnesium": "mg",
        "phosphate": "phos",
        "creatinine": "crea",
        "glucose": "glu",
        "lactate": "lact",
        "hemoglobin": "hgb",
        "platelets": "plt",
        "albumin": "alb",
        "blood urea nitrogen": "bun",
        "white blood cells": "wbc",
        "white blood cell count": "wbc",
        "bicarbonate": "bicar",
        "hco3": "bicar",  # Common alternative name
        "base excess": "be",
        "base_excess": "be",
        "troponin": "tnt",
        "troponin t": "tnt",
        "creatine kinase": "ck",
        "ck-mb": "ckmb",
        "c-reactive protein": "crp",
        "inr": "inr_pt",
        "pt": "inr_pt",  # Map pt to inr_pt (correct data_util name)
        "fibrinogen": "fgn",
        "neutrophils": "neut",
        "lymphocytes": "lymph",
        "band neutrophils": "bnd",
        "bands": "bnd",
        "methemoglobin": "methb",
        "alkaline phosphatase": "alp",
        "alanine aminotransferase": "alt",
        "aspartate aminotransferase": "ast",
        "bilirubin": "bili",
        "total bilirubin": "bili",
        "direct bilirubin": "bili_dir",
        "urine output": "urine",
        # Common clinical blood gas abbreviations
        "pao2": "po2",  # Partial pressure of oxygen
        "paco2": "pco2",  # Partial pressure of carbon dioxide
    }


def get_priority_features_for_task(task_name: str) -> set:
    """Get comprehensive task-specific features for investigation step."""
    task_features = {
        "mortality": {
            "map",
            "sbp",
            "dbp",
            "lact",
            "o2sat",
            "po2",
            "pco2",
            "crea",
            "bun",
            "urine",
            "bili",
            "alt",
            "ast",
            "ph",
            "be",
            "bicar",
            "plt",
            "inr_pt",
            "hr",
            "temp",
        },
        "aki": {"crea", "urine", "bun", "map", "sbp", "na", "k"},
        "sepsis": {
            "temp",
            "hr",
            "resp",
            "map",
            "sbp",
            "o2sat",
            "po2",
            "pco2",
            "fio2",
            "plt",
            "bili",
            "crea",
            "wbc",
            "neut",
            "bnd",
            "crp",
            "lact",
        },
    }

    return task_features.get(task_name, set())


def get_clinical_group_aliases() -> dict:
    """Get common clinical aliases for feature groups."""
    # Define subset lists for easier reference
    liver_subset = [
        f
        for f in features_dyn_liver_kidney_list
        if f in ["alb", "alp", "alt", "ast", "bili", "bili_dir"]
    ]
    kidney_subset = [
        f for f in features_dyn_liver_kidney_list if f in ["bun", "crea", "urine"]
    ]
    hematology_basic = [
        f for f in features_dyn_hematology_immune_list if f not in ["crp", "methb"]
    ]

    return {
        # Blood gas variations
        tuple(features_dyn_bga_list): [
            "blood gas analysis",
            "blood_gas",
            "bloodgas",  # Short form
            "arterial blood gas",
            "abg",
        ],
        # Metabolic panel variations
        tuple(features_dyn_electrolytes_met_list): [
            "metabolic panel & electrolytes",
            "metabolic_panel",
            "metabolic panel",
            "metab",  # Short form
            "comprehensive metabolic panel",
            "cmp",
        ],
        tuple(features_dyn_electrolytes_met_list[:5]): [  # Basic metabolic panel
            "basic metabolic panel",
            "bmp",
        ],
        tuple(features_dyn_electrolytes_met_list[:7]): [  # Electrolytes only
            "electrolytes"
        ],
        # Liver/kidney function variations
        tuple(features_dyn_liver_kidney_list): [
            "liver & kidney function",
            "liver_kidney_function",
            "liver kidney function",
        ],
        tuple(liver_subset): [
            "liver function",
            "liver_function",
            "liver",  # Short form
        ],
        tuple(kidney_subset): [
            "kidney function",
            "kidney_function",
            "renal function",
            "renal_function",
            "kidney",  # Short form
            "renal",  # Short form
        ],
        # Hematology variations
        tuple(features_dyn_hematology_immune_list): ["hematology & immune response"],
        tuple(hematology_basic): [
            "hematology",
            "heme",  # Short form
            "cbc",
            "complete blood count",
            "blood count",
        ],
        # Cardiac variations
        tuple(features_dyn_cardiac_list): [
            "cardiac markers",
            "cardiac_markers",
            "cardiac",
            "heart markers",
        ],
        # Coagulation variations
        tuple(features_dyn_coag_list): ["coagulation", "coag", "clotting studies"],
    }
