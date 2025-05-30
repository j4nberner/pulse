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

...

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

...