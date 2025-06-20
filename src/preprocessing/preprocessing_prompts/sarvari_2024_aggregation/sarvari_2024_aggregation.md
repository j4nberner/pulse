## A systematic evaluation of the performance of GPT-4 and PaLM2 to diagnose comorbidities in MIMIC-IV patients

https://onlinelibrary.wiley.com/doi/full/10.1002/hcs2.79

Approach:

- Aggregate time-series of each independent feature and prompt model with:
  - Featurename (unit: g/dL): min=3.000, max=3.060, mean=3.055

### Example Prompt as stated in paper:

Suggest as many potential diagnoses as possible from the following patient data.
In addition, include previously diagnosed conditions and information about patient's medical history (if any).
Give exact numbers and/or text quotes from the data that made you think of each of the diagnoses and, if necessary, give further tests that could confirm the diagnosis.
Once you're done, suggest further, more complex diseases that may be ongoing based on the existing diagnoses you already made.
Use the International Classification of Disease (ICD) standard for reporting the diagnoses.
Before finalizing your answer check if you haven't missed any abnormal data points and hence any diagnoses that could be made based on them. If you did, add them to your list of diagnoses

For example, if the patient data mentions:

“Blood report:
min glucose: 103, max glucose: 278, avg glucose: 156.5, max inr: 2.1, max pt: 22.4, max ptt: 150, avg wbc: 13.8, max wbc: 14.1, max lactate: 5.9, max bun: 101, max creatinine: 5.8, avg bun: 38.15, avg creatinine: 2.78
Blood gas report:
3 h after admission the blood gas results from venous blood are: ph: 7.2
Imaging report:
Status post left total shoulder replacement
Chest X-Ray Possible small right pleural effusion and Mild, bibasilar atelectasis… Lung volumes have slightly increased but areas of atelectasis are seen at both the left and the right lung bases
Microbiology tests:
24 h after admission the microbiology culture test MRSA SCREEN obtained via MRSA SCREEN identified POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS
Vitalsigns data from ICU:
max temperature: 38, min peripheral oxygen saturation: 70, max respiration rate: 29”

then your answer may be:

1: Methicillin resistant Staphylococcus aureus infection, site unspecified
Foundational data: Microbiology culture test identifying “POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS”
2: Atelectasis
Foundational data from Chest X-Ray: “Mild, bibasilar atelectasis… Lung volumes have slightly increased but areas of atelectasis are seen at both the left and the right lung bases”
3: Pleural effusion, not elsewhere classified
Foundational data from Chest X-Ray: “Possible small right pleural effusion.”
Further tests: Thoracentesis, CT chest
4: Acidosis
Foundational data: “ph: 7.2”
Further tests: Urine pH, Anion Gap
5: Lactic acidosis
Foundational data: “max lactate: 5.9”
6: Acquired coagulation factor deficienc
Foundational data: “max inr: 2.1, max pt: 22.4, max ptt: 150”
Further tests: Antiphospholipid Antibodies (APL), Protein C, Protein S, Antithrombin III, Factor V Leiden, Fibrinogen test
7: Hyperglycemia, unspecified
Foundational data: “max glucose: 278, avg glucose: 156.5.”
Further tests: Hemoglobin A1c (HbA1c) test
8: Hypoxemia
Foundational data: “min peripheral oxygen saturation: 70”
Further tests: Measure PaO2 in blood
9: Leukocytosis
Foundational data: “max wbc: 14.1, avg wbc: 13.8.” The patient's white blood cell count is consistently elevated which may suggest an ongoing inflammatory response or infection.
Further tests: Infection markers such as CRP or PCT, Assessment of symptoms like fever, inflammation or fatigue. 10. Unspecified acute kidney failure:
Foundational data: “max bun: 101, max creatinine: 5.8, avg bun: 38.15, avg creatinine: 2.78”
Further tests: Urine output measurements for oliguria, ultrasound to rule out obstruction 11. Presence of left artificial shoulder joint
Foundational data: The imaging report mentions: “Status post left total shoulder replacement”
Further diseases based on these diagnoses (continued the indexing from the previous number in the list):
12: Unspecified septicemia
Foundational data: positive MRSA screen, systemic inflammatory response: “max respiration rate: 29,” “max temperature: 38,” leukocytosis
Further tests: HR, BP, wound culture, respiratory excretion tests
13: Septic shock
Foundational data: Septicemia with acidosis and lactic acidosis may suggest septic shock
Further tests: patient examination (low BP, mental disorientation, nausea, pale skin may confirm the finding)
14: Acute respiratory failure, with hypoxia or hypercapnia
Foundational data: hypoxemia and the presence of atelectasis
Further tests: Clinical symptoms (severe shortness of breath, rapid breathing, and confusion), arterial blood gas measurements showing hypoxia or hypercapnia
15: Type 2 diabetes mellitus with diabetic chronic kidney disease
Foundational data: Hyperglycemia and kidney failure
Further tests: urine test, hemoglobin (A1C) test, GFR, BP, physical examination (swelling, nausea, weakness, and eye disease)

Patient data:

### Adjusted prompt for Pulse benchmark

Suggest a diagnosis of aki for the following patient data. Reply with aki or not-aki.
Give exact numbers and/or text quotes from the data that made you think of each of the diagnoses.
Before finalizing your answer check if you haven't missed any abnormal data points.

Patient data:

Patient Info — index: 0, age: 55.0, sex: Male, height: 170.0, weight: 95.0
Albumin (unit: g/dL): min=2.250, max=2.250, mean=2.250
Alkaline Phosphatase (unit: U/L): min=104.300, max=104.300, mean=104.300
Alanine Aminotransferase (ALT) (unit: U/L): min=24.000, max=151.020, mean=45.170
Aspartate Aminotransferase (AST) (unit: U/L): min=25.000, max=231.810, mean=59.468
Base Excess (unit: mmol/L): min=-1.200, max=-0.440, mean=-0.973
Bicarbonate (unit: mmol/L): min=23.400, max=24.600, mean=23.887
Total Bilirubin (unit: mg/dL): min=1.111, max=1.700, mean=1.443
Band Neutrophils (unit: %): min=21.360, max=21.360, mean=21.360
Blood Urea Nitrogen (BUN) (unit: mg/dL): min=17.640, max=22.760, mean=18.493
Calcium (unit: mg/dL): min=8.180, max=8.180, mean=8.180
Ionized Calcium (unit: mmol/L): min=1.160, max=1.210, mean=1.202
Creatine Kinase (CK) (unit: U/L): min=145.000, max=1169.990, mean=318.832
Creatine Kinase-MB (CK-MB) (unit: ng/mL): min=3.800, max=30.540, mean=8.457
Chloride (unit: mmol/L): min=105.000, max=108.150, mean=105.525
Creatinine (unit: mg/dL): min=0.769, max=1.000, mean=0.808
C-Reactive Protein (CRP) (unit: mg/L): min=7.000, max=93.640, mean=21.440
Diastolic Blood Pressure (unit: mmHg): min=48.000, max=82.000, mean=61.000
Fibrinogen (unit: mg/dL): min=279.250, max=321.000, mean=293.167
Fraction of Inspired Oxygen (FiO2) (unit: %): min=44.900, max=44.900, mean=44.900
Glucose (unit: mg/dL): min=152.020, max=162.144, mean=160.457
Hemoglobin (unit: g/dL): min=10.410, max=15.000, mean=14.235
Heart Rate (unit: bpm): min=84.000, max=100.000, mean=92.083
inr (unit: ): min=1.240, max=1.240, mean=1.240
Potassium (unit: mmol/L): min=4.100, max=4.600, mean=4.383
Lactate (unit: mmol/L): min=0.900, max=1.930, mean=1.305
Lymphocytes (unit: %): min=0.180, max=0.180, mean=0.180
Mean Arterial Pressure (MAP) (unit: mmHg): min=70.000, max=134.000, mean=88.833
Mean Corpuscular Hemoglobin (MCH) (unit: pg): min=30.720, max=31.000, mean=30.953
Mean Corpuscular Hemoglobin Concentration (MCHC) (unit: g/dL): min=33.850, max=35.100, mean=34.892
Mean Corpuscular Volume (MCV) (unit: fL): min=89.000, max=90.780, mean=89.297
Methemoglobin (unit: %): min=0.700, max=1.500, mean=1.008
Magnesium (unit: mg/dL): min=1.993, max=2.060, mean=2.005
Sodium (unit: mmol/L): min=133.000, max=137.310, mean=135.218
Neutrophils (unit: %): min=82.320, max=82.320, mean=82.320
Oxygen Saturation (unit: %): min=89.000, max=95.000, mean=92.833
Partial Pressure of Carbon Dioxide (PaCO2) (unit: mmHg): min=36.950, max=52.200, mean=49.658
pH Level (unit: /): min=7.312, max=7.420, mean=7.330
Phosphate (unit: mg/dL): min=3.360, max=5.173, mean=4.871
Platelets (unit: 1000/µL): min=179.480, max=191.000, mean=189.080
Partial Pressure of Oxygen (PaO2) (unit: mmHg): min=61.300, max=109.720, mean=69.370
Partial Thromboplastin Time (PTT) (unit: sec): min=33.300, max=46.150, mean=41.867
Respiratory Rate (unit: breaths/min): min=12.000, max=23.000, mean=17.333
Systolic Blood Pressure (unit: mmHg): min=127.000, max=170.000, mean=143.000
Temperature (unit: °C): min=35.700, max=36.700, mean=36.367
Troponin T (unit: ng/mL): min=0.004, max=0.980, mean=0.168
Urine Output (unit: mL/h): min=53.620, max=263.260, mean=88.560
White Blood Cell Count (WBC) (unit: 1000/µL): min=10.980, max=14.600, mean=13.997
