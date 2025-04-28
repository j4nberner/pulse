# Dictionary of feature limits for outlier detection
from typing import Tuple

features_dict = {
    "hr": ("Heart Rate [bpm]", (60, 100), (20, 320)),
    "sbp": ("Systolic Blood Pressure\n[mmHg]", (90, 120), (30, 300)),
    "dbp": ("Diastolic Blood Pressure\n[mmHg]", (60, 80), (10, 200)),
    "map": ("Mean Arterial Pressure (MAP) [mmHg]", (65, 100), (20, 250)),
    "o2sat": ("Oxygen Saturation [%]", (95, 100), (50, 100)),
    "resp": ("Respiratory Rate\n[breaths/min]", (12, 20), (4, 80)),
    "temp": ("Temperature [°C]", (36.5, 37.5), (30, 42)),
    "ph": ("pH Level [-]", (7.35, 7.45), (6.7, 8.0)),
    "po2": ("Partial Pressure of\nOxygen (PaO2) [mmHg]", (75, 100), (40, 600)),
    "pco2": ("Partial Pressure of\nCarbon Dioxide (PaCO2) [mmHg]", (35, 45), (10, 150)),
    "be": ("Base Excess [mmol/L]", (-2, 2), (-25, 25)),
    "bicar": ("Bicarbonate [mmol/L]", (22, 29), (5, 50)),
    "fio2": ("Fraction of Inspired Oxygen\n(FiO2) [%]", (21, 100), (21, 100)),
    "inr_pt": ("International Normalised Ratio\n(INR) [-]", (0.8, 1.2), (0.5, 20)),
    "ptt": ("Partial Thromboplastin Time\n(PTT) [sec]", (25, 35), (10, 250)),
    "fgn": ("Fibrinogen [mg/dL]", (200, 400), (30, 1100)),
    "na": ("Sodium [mmol/L]", (135, 145), (90, 170)),
    "k": ("Potassium [mmol/L]", (3.5, 5), (1, 9)),
    "cl": ("Chloride [mmol/L]", (96, 106), (70, 140)),
    "ca": ("Calcium [mg/dL]", (8.5, 10.5), (4, 20)),
    "cai": ("Ionized Calcium [mmol/L]", (1.1, 1.3), (0.4, 2.2)),
    "mg": ("Magnesium [mg/dL]", (1.7, 2.2), (0.5, 5)),
    "phos": ("Phosphate [mg/dL]", (2.5, 4.5), (0.5, 15)),
    "glu": ("Glucose [mg/dL]", (70, 140), (25, 1000)),
    "lact": ("Lactate [mmol/L]", (0.5, 2), (0.1, 20)),
    "alb": ("Albumin [g/dL]", (3.5, 5), (0.5, 6)),
    "alp": ("Alkaline Phosphatase [U/L]", (44, 147), (10, 1200)),
    "alt": ("Alanine Aminotransferase\n(ALT) [U/L]", (7, 56), (10, 5000)),
    "ast": ("Aspartate Aminotransferase\n(AST) [U/L]", (10, 40), (10, 8000)),
    "bili": ("Total Bilirubin [mg/dL]", (0.1, 1.2), (0.1, 50)),
    "bili_dir": ("Direct Bilirubin [mg/dL]", (0, 0.3), (0, 30)),
    "bun": ("Blood Urea Nitrogen\n(BUN) [mg/dL]", (7, 20), (1, 180)),
    "crea": ("Creatinine [mg/dL]", (0.6, 1.3), (0.1, 20)),
    "urine": ("Urine Output [mL/h]", (30, 50), (0, 2000)),
    "hgb": ("Hemoglobin [g/dL]", (13.5, 17.5), (3, 20)),
    "mch": ("Mean Corpuscular\nHemoglobin (MCH) [pg]", (27, 33), (15, 45)),
    "mchc": (
        "Mean Corpuscular Hemoglobin\nConcentration (MCHC) [g/dL]",
        (32, 36),
        (20, 45),
    ),
    "mcv": ("Mean Corpuscular\nVolume (MCV) [fL]", (80, 100), (50, 130)),
    "plt": ("Platelets [10^3/µL]", (150, 450), (10, 1500)),
    "wbc": ("White Blood Cell Count\n(WBC) [10^3/µL]", (4, 11), (0.1, 500)),
    "neut": ("Neutrophils [%]", (55, 70), (0, 100)),
    "bnd": ("Band Neutrophils [%]", (0, 6), (0, 50)),
    "lymph": ("Lymphocytes [%]", (20, 40), (0, 90)),
    "crp": ("C-Reactive Protein\n(CRP) [mg/L]", (0, 10), (0, 500)),
    "methb": ("Methemoglobin [%]", (0, 2), (0, 60)),
    "ck": ("Creatine Kinase\n(CK) [U/L]", (30, 200), (10, 100000)),
    "ckmb": ("Creatine Kinase-MB\n(CK-MB) [ng/mL]", (0, 5), (0, 500)),
    "tnt": ("Troponin T [ng/mL]", (0, 14), (0, 1000)),
    "height": ("Height [cm]", (), (135, 220)),
    "weight": ("Weight [kg]", (), (40, 250)),
}


def get_feature_name(feature_name: str) -> str:
    """Returns the feature name for a given feature key."""
    return features_dict.get(feature_name, (feature_name, (0, 0), (0, 0)))[0]


def get_feature_limits(
    feature_name: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Returns the feature limits for a given feature key."""
    return features_dict.get(feature_name, (feature_name, (0, 0), (0, 0)))[1:]


def get_feature_name_and_limits(
    feature_name: str,
) -> Tuple[str, Tuple[float, float], Tuple[float, float]]:
    """Returns the feature name and limits for a given feature key."""
    return features_dict.get(feature_name, (feature_name, (0, 0), (0, 0)))
