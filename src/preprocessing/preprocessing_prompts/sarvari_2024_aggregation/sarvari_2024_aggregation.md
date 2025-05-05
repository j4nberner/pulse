## A systematic evaluation of the performance of GPT-4 and PaLM2 to diagnose comorbidities in MIMIC-IV patients

https://onlinelibrary.wiley.com/doi/full/10.1002/hcs2.79

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
Further tests: Infection markers such as CRP or PCT, Assessment of symptoms like fever, inflammation or fatigue.
10. Unspecified acute kidney failure:
    Foundational data: “max bun: 101, max creatinine: 5.8, avg bun: 38.15, avg creatinine: 2.78”
Further tests: Urine output measurements for oliguria, ultrasound to rule out obstruction
11. Presence of left artificial shoulder joint
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

Suggest a diagnoses of mortality or not-mortality for the following patient data.
In addition, include information about patient's medical history (if any). 
Give exact numbers and/or text quotes from the data that made you think of each of the diagnoses and, if necessary, give further tests that could confirm the diagnosis.
Use the International Classification of Disease (ICD) standard for reporting the diagnoses.
Before finalizing your answer check if you haven't missed any abnormal data points. ´´´

For example, if the patient data mentions:

Patient Info — index: 674, stay_id: 30128094, age: 65.0, sex: Female, weight: 57.6, height: 169.44
Albumin (unit: g/dL): min=3.000, max=3.060, mean=3.055
Alkaline Phosphatase (unit: U/L): min=52.000, max=119.480, mean=114.082
Alanine Aminotransferase (ALT) (unit: U/L): min=22.000, max=189.700, mean=176.284
Aspartate Aminotransferase (AST) (unit: U/L): min=27.000, max=300.580, mean=278.694
Base Excess (unit: mmol/L): min=-6.000, max=-1.610, mean=-3.313
Bicarbonate (unit: mmol/L): min=21.000, max=22.750, mean=21.570
Total Bilirubin (unit: mg/dL): min=0.400, max=3.300, mean=2.765
Band Neutrophils (unit: %): min=5.710, max=5.710, mean=5.710
Blood Urea Nitrogen (BUN) (unit: mg/dL): min=12.000, max=28.900, mean=19.932
Calcium (unit: mg/dL): min=7.300, max=8.290, mean=7.633
Ionized Calcium (unit: mmol/L): min=1.130, max=1.130, mean=1.130
Creatine Kinase (CK) (unit: U/L): min=1553.340, max=1553.340, mean=1553.340
Creatine Kinase-MB (CK-MB) (unit: ng/mL): min=25.030, max=25.030, mean=25.030
Chloride (unit: mmol/L): min=102.000, max=105.000, mean=104.483
Creatinine (unit: mg/dL): min=0.900, max=1.580, mean=1.090
C-Reactive Protein (CRP) (unit: mg/L): min=77.230, max=77.230, mean=77.230
Diastolic Blood Pressure (unit: mmHg): min=44.000, max=70.000, mean=58.730
Fibrinogen (unit: mg/dL): min=265.640, max=720.000, mean=449.419
Fraction of Inspired Oxygen (FiO2) (unit: %): min=53.970, max=53.970, mean=53.970
Glucose (unit: mg/dL): min=105.000, max=145.910, mean=120.295
Hemoglobin (unit: g/dL): min=9.900, max=10.200, mean=10.000
Heart Rate (unit: bpm): min=65.000, max=98.000, mean=79.800
inr (unit: ): min=1.100, max=1.550, mean=1.482
Potassium (unit: mmol/L): min=4.230, max=4.400, mean=4.344
Lactate (unit: mmol/L): min=0.700, max=2.720, mean=1.358
Lymphocytes (unit: %): min=12.460, max=12.460, mean=12.460
Mean Arterial Pressure (MAP) (unit: mmHg): min=59.000, max=83.000, mean=66.620
Mean Corpuscular Hemoglobin (MCH) (unit: pg): min=30.060, max=31.400, mean=31.009
Mean Corpuscular Hemoglobin Concentration (MCHC) (unit: g/dL): min=32.300, max=33.050, mean=32.638
Mean Corpuscular Volume (MCV) (unit: fL): min=91.040, max=97.000, mean=95.331
Methemoglobin (unit: %): min=1.080, max=1.080, mean=1.080
Magnesium (unit: mg/dL): min=2.070, max=2.900, mean=2.644
Sodium (unit: mmol/L): min=135.000, max=138.190, mean=135.973
Neutrophils (unit: %): min=77.710, max=77.710, mean=77.710
Oxygen Saturation (unit: %): min=93.000, max=100.000, mean=97.520
Partial Pressure of Carbon Dioxide (PaCO2) (unit: mmHg): min=39.000, max=55.000, mean=41.912
pH Level (unit: /): min=7.200, max=7.360, mean=7.327
Phosphate (unit: mg/dL): min=2.800, max=3.770, mean=3.456
Platelets (unit: 1000/µL): min=167.000, max=191.280, mean=176.998
Partial Pressure of Oxygen (PaO2) (unit: mmHg): min=98.000, max=155.160, mean=107.099
Partial Thromboplastin Time (PTT) (unit: sec): min=32.900, max=41.260, mean=37.481
Respiratory Rate (unit: breaths/min): min=3.000, max=19.000, mean=10.900
Systolic Blood Pressure (unit: mmHg): min=89.000, max=119.000, mean=108.510
Temperature (unit: °C): min=36.722, max=37.222, mean=36.976
Troponin T (unit: ng/mL): min=0.940, max=0.940, mean=0.940
Urine Output (unit: mL/h): min=50.000, max=400.000, mean=168.430
White Blood Cell Count (WBC) (unit: 1000/µL): min=12.810, max=17.500, mean=16.139

Then your answer may be: 
{
 "diagnosis": "not-mortality",
  "probability": "the probability of your estimation as a float",
  "explanation": "a brief explanation for the prediction"
}



Patient data:

Patient Info — index: 0, stay_id: 38989889, age: 81.0, sex: Male, weight: 122.3, height: 185.42000000000002
Albumin (unit: g/dL): min=3.060, max=3.060, mean=3.060
Alkaline Phosphatase (unit: U/L): min=119.480, max=119.480, mean=119.480
Alanine Aminotransferase (ALT) (unit: U/L): min=189.700, max=189.700, mean=189.700
Aspartate Aminotransferase (AST) (unit: U/L): min=300.580, max=300.580, mean=300.580
Base Excess (unit: mmol/L): min=-3.000, max=0.000, mean=-0.640
Bicarbonate (unit: mmol/L): min=20.000, max=23.000, mean=20.960
Total Bilirubin (unit: mg/dL): min=2.390, max=3.300, mean=2.845
Band Neutrophils (unit: %): min=5.710, max=5.710, mean=5.710
Blood Urea Nitrogen (BUN) (unit: mg/dL): min=25.000, max=34.000, mean=31.120
Calcium (unit: mg/dL): min=8.800, max=9.100, mean=9.004
Ionized Calcium (unit: mmol/L): min=1.180, max=1.210, mean=1.200
Creatine Kinase (CK) (unit: U/L): min=1553.340, max=1553.340, mean=1553.340
Creatine Kinase-MB (CK-MB) (unit: ng/mL): min=25.030, max=25.030, mean=25.030
Chloride (unit: mmol/L): min=99.000, max=106.000, mean=101.240
Creatinine (unit: mg/dL): min=1.100, max=1.500, mean=1.372
C-Reactive Protein (CRP) (unit: mg/L): min=77.230, max=77.230, mean=77.230
Diastolic Blood Pressure (unit: mmHg): min=42.000, max=56.000, mean=47.980
Fibrinogen (unit: mg/dL): min=597.000, max=597.000, mean=597.000
Fraction of Inspired Oxygen (FiO2) (unit: %): min=50.000, max=100.000, mean=56.400
Glucose (unit: mg/dL): min=105.000, max=209.000, mean=171.440
Hemoglobin (unit: g/dL): min=9.000, max=9.200, mean=9.136
Heart Rate (unit: bpm): min=54.000, max=103.000, mean=77.040
inr (unit: ): min=2.100, max=2.300, mean=2.164
Potassium (unit: mmol/L): min=4.200, max=4.200, mean=4.200
Lactate (unit: mmol/L): min=1.100, max=2.100, mean=1.524
Lymphocytes (unit: %): min=12.460, max=12.460, mean=12.460
Mean Arterial Pressure (MAP) (unit: mmHg): min=58.000, max=77.000, mean=66.380
Mean Corpuscular Hemoglobin (MCH) (unit: pg): min=29.000, max=29.500, mean=29.160
Mean Corpuscular Hemoglobin Concentration (MCHC) (unit: g/dL): min=32.100, max=32.300, mean=32.236
Mean Corpuscular Volume (MCV) (unit: fL): min=90.000, max=92.000, mean=90.640
Methemoglobin (unit: %): min=1.080, max=1.080, mean=1.080
Magnesium (unit: mg/dL): min=1.700, max=2.200, mean=1.860
Sodium (unit: mmol/L): min=134.000, max=141.000, mean=136.240
Neutrophils (unit: %): min=77.710, max=77.710, mean=77.710
Oxygen Saturation (unit: %): min=90.000, max=100.000, mean=95.840
Partial Pressure of Carbon Dioxide (PaCO2) (unit: mmHg): min=39.000, max=45.000, mean=41.040
pH Level (unit: /): min=7.320, max=7.410, mean=7.378
Phosphate (unit: mg/dL): min=2.900, max=3.800, mean=3.512
Platelets (unit: 1000/µL): min=122.000, max=156.000, mean=145.120
Partial Pressure of Oxygen (PaO2) (unit: mmHg): min=94.000, max=233.000, mean=137.480
Partial Thromboplastin Time (PTT) (unit: sec): min=33.300, max=33.900, mean=33.492
Respiratory Rate (unit: breaths/min): min=8.000, max=22.000, mean=16.820
Systolic Blood Pressure (unit: mmHg): min=95.000, max=128.000, mean=108.020
Temperature (unit: °C): min=35.500, max=37.111, mean=36.533
Troponin T (unit: ng/mL): min=0.940, max=0.940, mean=0.940
Urine Output (unit: mL/h): min=35.000, max=325.000, mean=151.615
White Blood Cell Count (WBC) (unit: 1000/µL): min=9.900, max=16.000, mean=14.048