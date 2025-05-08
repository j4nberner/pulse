## Large Language Models are Few-Shot Health Learners

Paper: https://arxiv.org/pdf/2305.15525

---

### Prompt used in paper:

Question:
Classify the given interbeat interval sequence in ms as either Atrial Fibrillation or Normal.
Sinus: 896,1192,592,1024,1072,808,888,896,760,1000,784,736,856,1000,1272,824,
872,1120,840,896,888,560,1248,824,968,960,1000,1008,776,744,896,1256.

Answer: "

### Adapted prompt:

Example Question: Classify the following ICU patient data as either aki or not-aki

stay_id (unit: ): 2
age (unit: ): 80.0
sex (unit: ): Male
Height (unit: cm): 170.0
Weight (unit: kg): 60.0
Albumin (unit: g/dL): [2.25, 2.25, 2.25, 2.25, 2.25, 2.25]
Alkaline Phosphatase (unit: U/L): [104.3, 104.3, 104.3, 104.3, 104.3, 104.3]
Alanine Aminotransferase (ALT) (unit: U/L): [35.0, 35.0, 35.0, 35.0, 35.0, 35.0]
Aspartate Aminotransferase (AST) (unit: U/L): [62.0, 62.0, 62.0, 62.0, 62.0, 62.0]
Base Excess (unit: mmol/L): [-0.44, -0.44, -0.44, -0.44, -0.44, -0.44]
Bicarbonate (unit: mmol/L): [23.92, 23.92, 23.92, 23.92, 23.92, 23.92]
Total Bilirubin (unit: mg/dL): [2.221746, 2.221746, 2.221746, 2.221746, 2.221746, 2.221746]
Band Neutrophils (unit: %): [21.36, 21.36, 21.36, 21.36, 21.36, 21.36]
Blood Urea Nitrogen (BUN) (unit: mg/dL): [30.52, 30.52, 30.52, 30.52, 30.52, 30.52]
Calcium (unit: mg/dL): [8.17632, 8.17632, 8.17632, 8.17632, 8.17632, 8.17632]
Ionized Calcium (unit: mmol/L): [1.16, 1.16, 1.16, 1.16, 1.16, 1.16]
Creatine Kinase (CK) (unit: U/L): [1964.0, 1964.0, 1964.0, 1964.0, 1964.0, 1964.0]
Creatine Kinase-MB (CK-MB) (unit: ng/mL): [14.4, 14.4, 14.4, 14.4, 14.4, 14.4]
Chloride (unit: mmol/L): [104.0, 104.0, 104.0, 104.0, 104.0, 104.0]
Creatinine (unit: mg/dL): [0.927584, 0.927584, 0.927584, 0.927584, 0.927584, 0.927584]
C-Reactive Protein (CRP) (unit: mg/L): [34.0, 34.0, 34.0, 34.0, 34.0, 34.0]
Diastolic Blood Pressure (unit: mmHg): [57.71, 57.71, 57.71, 57.71, 57.71, 57.71]
Fibrinogen (unit: mg/dL): [279.25, 279.25, 279.25, 279.25, 279.25, 279.25]
Fraction of Inspired Oxygen (FiO2) (unit: %): [44.9, 44.9, 44.9, 44.9, 44.9, 44.9]
Glucose (unit: mg/dL): [156.73919999999998, 156.73919999999998, 156.73919999999998, 156.73919999999998, 156.73919999999998, 156.73919999999998]
Hemoglobin (unit: g/dL): [13.100000000000001, 13.100000000000001, 13.100000000000001, 13.100000000000001, 13.100000000000001, 13.100000000000001]
Heart Rate (unit: bpm): [58.5, 58.0, 58.0, 61.0, 64.0, 63.0]
Potassium (unit: mmol/L): [4.5, 4.5, 4.5, 4.5, 4.5, 4.5]
Lactate (unit: mmol/L): [1.93, 1.93, 1.93, 1.93, 1.93, 1.93]
Lymphocytes (unit: %): [0.18, 0.18, 0.18, 0.18, 0.18, 0.18]
Mean Arterial Pressure (MAP) (unit: mmHg): [99.0, 101.0, 91.0, 90.5, 81.0, 97.0]
Mean Corpuscular Hemoglobin (MCH) (unit: pg): [31.0, 31.0, 31.0, 31.0, 31.0, 31.0]
Mean Corpuscular Hemoglobin Concentration (MCHC) (unit: g/dL): [34.0, 34.0, 34.0, 34.0, 34.0, 34.0]
Mean Corpuscular Volume (MCV) (unit: fL): [92.0, 92.0, 92.0, 92.0, 92.0, 92.0]
Methemoglobin (unit: %): [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
Magnesium (unit: mg/dL): [2.40669, 2.40669, 2.40669, 2.40669, 2.40669, 2.40669]
Sodium (unit: mmol/L): [140.0, 140.0, 140.0, 140.0, 140.0, 140.0]
Neutrophils (unit: %): [82.32, 82.32, 82.32, 82.32, 82.32, 82.32]
Oxygen Saturation (unit: %): [93.0, 95.0, 95.0, 94.0, 95.0, 94.0]
Partial Pressure of Carbon Dioxide (PaCO2) (unit: mmHg): [36.95, 36.95, 36.95, 36.95, 36.95, 36.95]
pH Level (unit: /): [7.42, 7.42, 7.42, 7.42, 7.42, 7.42]
Phosphate (unit: mg/dL): [4.0887277200000005, 4.0887277200000005, 4.0887277200000005, 4.0887277200000005, 4.0887277200000005, 4.0887277200000005]
Platelets (unit: 1000/µL): [131.0, 131.0, 131.0, 131.0, 131.0, 131.0]
Partial Pressure of Oxygen (PaO2) (unit: mmHg): [109.72, 109.72, 109.72, 109.72, 109.72, 109.72]
Partial Thromboplastin Time (PTT) (unit: sec): [46.15, 46.15, 46.15, 46.15, 46.15, 46.15]
Respiratory Rate (unit: breaths/min): [18.5, 18.5, 17.0, 17.0, 12.0, 22.0]
Systolic Blood Pressure (unit: mmHg): [119.12, 119.12, 119.12, 119.12, 119.12, 119.12]
Temperature (unit: °C): [36.8, 36.8, 36.8, 36.4, 36.4, 36.4]
Troponin T (unit: ng/mL): [0.012, 0.012, 0.012, 0.012, 0.012, 0.012]
Urine Output (unit: mL/h): [304.03067, 371.9624, 38.36130000000003, 38.36130000000003, 38.36130000000003, 571.3387]
White Blood Cell Count (WBC) (unit: 1000/µL): [11.8, 11.8, 11.8, 11.8, 11.8, 11.8]

Answer:
{
"diagnosis": "not-aki",
"probability": "the probability of your estimation as a float",
"explanation": "a brief explanation for the prediction"
}

Question: Classify the following ICU patient data as either aki or not-aki

stay_id (unit: ): 30505
age (unit: ): 55.0
sex (unit: ): Male
Height (unit: cm): 170.0
Weight (unit: kg): 95.0
Albumin (unit: g/dL): [2.25, 2.25, 2.25, 2.25, 2.25, 2.25]
Alkaline Phosphatase (unit: U/L): [104.3, 104.3, 104.3, 104.3, 104.3, 104.3]
Alanine Aminotransferase (ALT) (unit: U/L): [151.02, 24.0, 24.0, 24.0, 24.0, 24.0]
Aspartate Aminotransferase (AST) (unit: U/L): [231.81, 25.0, 25.0, 25.0, 25.0, 25.0]
Base Excess (unit: mmol/L): [-0.44, -1.0, -1.0, -1.0, -1.2, -1.2]
Bicarbonate (unit: mmol/L): [23.92, 23.4, 23.4, 23.4, 24.6, 24.6]
Total Bilirubin (unit: mg/dL): [1.56, 1.110873, 1.110873, 1.110873, 1.110873, 1.110873]
Band Neutrophils (unit: %): [21.36, 21.36, 21.36, 21.36, 21.36, 21.36]
Blood Urea Nitrogen (BUN) (unit: mg/dL): [22.76, 17.639999999999997, 17.639999999999997, 17.639999999999997, 17.639999999999997, 17.639999999999997]
Calcium (unit: mg/dL): [8.18, 8.18, 8.18, 8.18, 8.18, 8.18]
Ionized Calcium (unit: mmol/L): [1.16, 1.21, 1.21, 1.21, 1.21, 1.21]
Creatine Kinase (CK) (unit: U/L): [1169.99, 151.0, 151.0, 151.0, 145.0, 145.0]
Creatine Kinase-MB (CK-MB) (unit: ng/mL): [30.54, 4.2, 4.2, 4.2, 3.8, 3.8]
Chloride (unit: mmol/L): [108.15, 105.0, 105.0, 105.0, 105.0, 105.0]
Creatinine (unit: mg/dL): [1.0, 0.7692159999999999, 0.7692159999999999, 0.7692159999999999, 0.7692159999999999, 0.7692159999999999]
C-Reactive Protein (CRP) (unit: mg/L): [93.64, 7.0, 7.0, 7.0, 7.0, 7.0]
Diastolic Blood Pressure (unit: mmHg): [82.0, 67.0, 63.0, 55.0, 51.0, 48.0]
Fibrinogen (unit: mg/dL): [279.25, 279.25, 279.25, 279.25, 321.0, 321.0]
Fraction of Inspired Oxygen (FiO2) (unit: %): [44.9, 44.9, 44.9, 44.9, 44.9, 44.9]
Glucose (unit: mg/dL): [152.02, 162.14399999999998, 162.14399999999998, 162.14399999999998, 162.14399999999998, 162.14399999999998]
Hemoglobin (unit: g/dL): [10.41, 15.0, 15.0, 15.0, 15.0, 15.0]
Heart Rate (unit: bpm): [93.0, 88.0, 84.0, 88.5, 99.0, 100.0]
Potassium (unit: mmol/L): [4.1, 4.6, 4.6, 4.6, 4.2, 4.2]
Lactate (unit: mmol/L): [1.93, 0.9, 0.9, 0.9, 1.6, 1.6]
Lymphocytes (unit: %): [0.18, 0.18, 0.18, 0.18, 0.18, 0.18]
Mean Arterial Pressure (MAP) (unit: mmHg): [134.0, 94.0, 86.0, 75.0, 74.0, 70.0]
Mean Corpuscular Hemoglobin (MCH) (unit: pg): [30.72, 31.0, 31.0, 31.0, 31.0, 31.0]
Mean Corpuscular Hemoglobin Concentration (MCHC) (unit: g/dL): [33.85, 35.1, 35.1, 35.1, 35.1, 35.1]
Mean Corpuscular Volume (MCV) (unit: fL): [90.78, 89.0, 89.0, 89.0, 89.0, 89.0]
Methemoglobin (unit: %): [0.95, 0.7, 0.7, 0.7, 1.5, 1.5]
Magnesium (unit: mg/dL): [2.06, 1.99342, 1.99342, 1.99342, 1.99342, 1.99342]
Sodium (unit: mmol/L): [137.31, 136.0, 136.0, 136.0, 133.0, 133.0]
Neutrophils (unit: %): [82.32, 82.32, 82.32, 82.32, 82.32, 82.32]
Oxygen Saturation (unit: %): [90.0, 89.0, 93.0, 95.0, 95.0, 95.0]
Partial Pressure of Carbon Dioxide (PaCO2) (unit: mmHg): [36.95, 52.2, 52.2, 52.2, 52.2, 52.2]
pH Level (unit: /): [7.42, 7.312, 7.312, 7.312, 7.312, 7.312]
Phosphate (unit: mg/dL): [3.36, 5.17286007, 5.17286007, 5.17286007, 5.17286007, 5.17286007]
Platelets (unit: 1000/µL): [179.48, 191.0, 191.0, 191.0, 191.0, 191.0]
Partial Pressure of Oxygen (PaO2) (unit: mmHg): [109.72, 61.3, 61.3, 61.3, 61.3, 61.3]
Partial Thromboplastin Time (PTT) (unit: sec): [46.15, 46.15, 46.15, 46.15, 33.3, 33.3]
Respiratory Rate (unit: breaths/min): [23.0, 23.0, 12.0, 12.0, 17.5, 16.5]
Systolic Blood Pressure (unit: mmHg): [170.0, 153.0, 143.0, 130.0, 135.0, 127.0]
Temperature (unit: °C): [35.7, 35.7, 36.7, 36.7, 36.7, 36.7]
Troponin T (unit: ng/mL): [0.98, 0.004, 0.004, 0.004, 0.007, 0.007]
Urine Output (unit: mL/h): [263.26, 53.619717, 53.619717, 53.619717, 53.619717, 53.619717]
White Blood Cell Count (WBC) (unit: 1000/µL): [10.98, 14.6, 14.6, 14.6, 14.6, 14.6]

---
