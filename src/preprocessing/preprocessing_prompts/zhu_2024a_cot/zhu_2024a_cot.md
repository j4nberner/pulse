# Prompting Large Language Models for Zero-Shot Clinical Prediction with Structured Longitudinal Electronic Health Record Data
Zhu et al. (02/2024)  
Paper: https://arxiv.org/pdf/2402.01713

<img src="./zhu_2024_prompt_elements.png" alt="Prompt Elements" style="width:60%;">

## Original best performing prompt template using Chain-of-Thought (CoT) prompting for mortality prediction on the MIMIC-IV dataset

<div style="color:blue;">

**Role**

You are an experienced doctor in Intensive Care Unit (ICU) treatment.

I will provide you with medical information from multiple Intensive Care Unit (ICU) visits of a patient, each characterized by a fixed number of features.

Present multiple visit data of a patient in one batch. Represent each feature within this data as a string of values, separated by commas.

**Instruction**

Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of the patient not surviving their hospital stay. Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death.

In situations where the data does not allow for a reasonable conclusion, respond with the phrase “I do not know” without any additional explanation.

**Clinical Context: Unit & Reference Range** 

- Capillary refill rate: Unit: /. Reference range: /.
- Glascow coma scale eye opening: Unit: /. Reference range: /.
- Glascow coma scale motor response: Unit: /. Reference range: /.
- Glascow coma scale total: Unit: /. Reference range: /.
- Glascow coma scale verbal response: Unit: /. Reference range: /.
- Diastolic blood pressure: Unit: mmHg. Reference range: less than 80.
- Fraction inspired oxygen: Unit: /. Reference range: more than 0.21.
- Glucose: Unit: mg/dL. Reference range: 70 - 100.
- Heart Rate: Unit: bpm. Reference range: 60 - 100.
- Height: Unit: cm. Reference range: /.
- Mean blood pressure: Unit: mmHg. Reference range: less than 100.
- Oxygen saturation: Unit: %. Reference range: 95 - 100.
- Respiratory rate: Unit: breaths per minute. Reference range: 15 - 18.
- Systolic blood pressure: Unit: mmHg. Reference range: less than 120.
- Temperature: Unit: degrees Celsius. Reference range: 36.1 - 37.2.
- Weight: Unit: kg. Reference range: /.
- pH: Unit: /. Reference range: 7.35 - 7.45.

**Clinical Context: Example**

Here is an example of input information:  
Example #1:  
Input information of a patient
The patient is a female, aged 52 years.  
The patient had 4 visits that occurred at 0, 1, 2, 3.  
Details of the features for each visit are as follows:

- Capillary refill rate: "unknown, unknown, unknown, unknown"
- Glascow coma scale eye opening: "Spontaneously, Spontaneously, Spontaneously,
Spontaneously"
- Glascow coma scale motor response: "Obeys Commands, Obeys Commands, Obeys Commands,
Obeys Commands"
- …

RESPONSE:
0.3

**Input Data**

Input information of a patient:  
The patient is a male, aged 50.0 years.  
The patient had 4 visits that occurred at 0, 1, 2, 3.  
Details of the features for each visit are as follows:

- Capillary refill rate: "unknown, unknown, unknown, unknown"
- Glascow coma scale eye opening: "None, None, None, None"
- Glascow coma scale motor response: "Flex-withdraws, Flex-withdraws, unknown, Localizes Pain"
- Glascow coma scale total: "unknown, unknown, unknown, unknown"
- Glascow coma scale verbal response: "No Response-ETT, No Response-ETT, No Response-ETT, No Response-ETT"
- Diastolic blood pressure: "79.41666666666667, 77.83333333333333, 85.83333333333333, 83.25"
- Fraction inspired oxygen: "0.5, 0.5, 0.5, 0.5"
- Glucose: "172.0, 150.0, 128.0, 147.0"
- Heart Rate: "85.41666666666667, 84.91666666666667, 87.33333333333333, 88.41666666666667"
- Height: "173.0, 173.0, 173.0, 173.0"
- Mean blood pressure: "96.41666666666667, 97.58333333333333, 109.91666666666667, 108.41666666666667"
- Oxygen saturation: "99.5, 100.0, 100.0, 100.0"
- Respiratory rate: "22.083333333333332, 21.583333333333332, 22.0, 22.25"
- Systolic blood pressure: "136.5, 135.41666666666666, 152.16666666666666, 153.16666666666666"
- Temperature: "37.31944444444444, 37.074074074074076, 37.36296296296297, 37.76666666666667"
- Weight: "69.4, 69.4, 68.33174919999999, 68.2"
- pH: "7.484999999999999, 7.484999999999999, 7.45, 7.43"

**Output Indicator**

Please follow the Chain-of-Thought Analysis Process:
1. Analyze the data step by step, For example:
    - Blood pressure shows a slight downward trend, indicating...
    - Heart rate is stable, suggesting...
    - Lab results indicate [specific condition or lack thereof]...
    - The patient underwent [specific intervention], which could mean...

2. Make Intermediate Conclusions:
    - Draw intermediate conclusions from each piece of data. For example:
        - If a patient’s blood pressure is consistently low, it might indicate poor cardiovascular function.
        - The patient’s cardiovascular function is [conclusion].
        - [Other intermediate conclusions based on data].

3. Aggregate the Findings:
    - After analyzing each piece of data, aggregate these findings to form a comprehensive view of the patient’s condition.
    - Summarize key points from the initial analysis and intermediate conclusions.

    Aggregated Findings:
    - Considering the patient’s vital signs and lab results, the overall health status is...

4. Final Assessment:
    - Conclude with an assessment of the patient’s likelihood of not surviving their hospital stay.
    - Provide a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death.
    - If the data is insufficient or ambiguous, conclude with "I do not know."

    [0.XX] or "I do not know."

Example Chain-of-Thought Analysis:

1. Analyze the data step by step:
    - Blood pressure shows a slight downward trend, which might indicate a gradual decline in cardiovascular stability.
    - Heart rate is stable, which is a good sign, suggesting no immediate cardiac distress.
    - Elevated white blood cell count could indicate an infection or an inflammatory process in the body.
    - Low potassium levels might affect heart rhythm and overall muscle function.

2. Make Intermediate Conclusions:
    - The decreasing blood pressure could be a sign of worsening heart function or infection-related hypotension.
    - Stable heart rate is reassuring but does not completely rule out underlying issues.
    - Possible infection, considering the elevated white blood cell count.
    - Potassium levels need to be corrected to prevent cardiac complications.

3. Aggregate the Findings:
    - The patient is possibly facing a cardiovascular challenge, compounded by an infection and electrolyte imbalance.

    Aggregated Findings:
    - Considering the downward trend in blood pressure, stable heart rate, signs of infection, and electrolyte imbalance, the patient’s overall health status seems to be moderately compromised.

4. Final Assessment:  
0.65

Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death. Do not include any additional explanation.  
RESPONSE:
</div>

## Modified version of the prompt for use in the PULSE benchmark

You are an experienced doctor in Intensive Care Unit (ICU) treatment.

I will provide you with medical information from an Intensive Care Unit (ICU) visit of a patient, characterized by a fixed number of features.

<span style="color:red;">The data for multiple hours of the patient’s ICU stay will be presented in a one batch. Each feature within this data is represented</span> as a string of values separated by commas.

Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of the patient not surviving their hospital stay<span style="color:red;">/having acute kidney injury/sepsis at the end of the data batch</span>.

- Heart Rate: Unit: bpm. Reference range: 60 - 100.
- Systolic Blood Pressure: Unit: mmHg. Reference range: <span style="color:red;">90 -</span> 120.
- Diastolic Blood Pressure: Unit: mmHg. Reference range: <span style="color:red;">60 -</span> 80.
- Mean <span style="color:red;">Arterial</span> Pressure <span style="color:red;">(MAP)</span>: Unit: mmHg. Reference range: <span style="color:red;">65 - 100</span>.
- Oxygen Saturation: Unit: <span style="color:red;">%</span>. Reference range: <span style="color:red;">95 - 100</span>.
- Respiratory Rate: Unit: breaths/min. Reference range: <span style="color:red;">12 - 20</span>.
- Temperature: Unit: <span style="color:red;">°C</span>. Reference range: 36.<span style="color:red;">5</span> - 37.<span style="color:red;">5</span>.
- pH <span style="color:red;">Level</span>: Unit: /. Reference range: 7.35 - 7.45.
- <span style="color:red;">Partial Pressure of Oxygen (PaO2): Unit: mmHg. Reference range: 75 - 100.
- <span style="color:red;">Partial Pressure of Carbon Dioxide (PaCO2): Unit: mmHg. Reference range: 35 - 45.
- <span style="color:red;">Base Excess: Unit: mmol/L. Reference range: -2 - 2.
- <span style="color:red;">Bicarbonate: Unit: mmol/L. Reference range: 22 - 29.
- Fraction of Inspired Oxygen <span style="color:red;">(FiO2)</span>: Unit: <span style="color:red;">%</span>. Reference range: 21 <span style="color:red;">- 100</span>.
- <span style="color:red;">International Normalized Ratio (INR): Unit: /. Reference range: 0.8 - 1.2.
- <span style="color:red;">Partial Thromboplastin Time (PTT): Unit: sec. Reference range: 25 - 35.
- <span style="color:red;">Fibrinogen: Unit: mg/dL. Reference range: 200 - 400.
- <span style="color:red;">Sodium: Unit: mmol/L. Reference range: 135 - 145.
- <span style="color:red;">Potassium: Unit: mmol/L. Reference range: 3.5 - 5.
- <span style="color:red;">Chloride: Unit: mmol/L. Reference range: 96 - 106.
- <span style="color:red;">Calcium: Unit: mg/dL. Reference range: 8.5 - 10.5.
- <span style="color:red;">Ionized Calcium: Unit: mmol/L. Reference range: 1.1 - 1.3.
- <span style="color:red;">Magnesium: Unit: mg/dL. Reference range: 1.7 - 2.2.
- <span style="color:red;">Phosphate: Unit: mg/dL. Reference range: 2.5 - 4.5.
- Glucose: Unit: mg/dL. Reference range: 70 - 1<span style="color:red;">40</span>.
- <span style="color:red;">Lactate: Unit: mmol/L. Reference range: 0.5 - 2.
- <span style="color:red;">Albumin: Unit: g/dL. Reference range: 3.5 - 5.
- <span style="color:red;">Alkaline Phosphatase: Unit: U/L. Reference range: 44 - 147.
- <span style="color:red;">Alanine Aminotransferase (ALT): Unit: U/L. Reference range: 7 - 56.
- <span style="color:red;">Aspartate Aminotransferase (AST): Unit: U/L. Reference range: 10 - 40.
- <span style="color:red;">Total Bilirubin: Unit: mg/dL. Reference range: 0.1 - 1.2.
- <span style="color:red;">Direct Bilirubin: Unit: mg/dL. Reference range: 0 - 0.3.
- <span style="color:red;">Blood Urea Nitrogen (BUN): Unit: mg/dL. Reference range: 7 - 20.
- <span style="color:red;">Creatinine: Unit: mg/dL. Reference range: 0.6 - 1.3.
- <span style="color:red;">Urine Output: Unit: mL/h. Reference range: 30 - 50.
- <span style="color:red;">Hemoglobin: Unit: g/dL. Reference range: 12.5 - 17.5.
- <span style="color:red;">Mean Corpuscular Hemoglobin (MCH): Unit: pg. Reference range: 27 - 33.
- <span style="color:red;">Mean Corpuscular Hemoglobin Concentration (MCHC): Unit: g/dL. Reference range: 32 - 36.
- <span style="color:red;">Mean Corpuscular Volume (MCV): Unit: fL. Reference range: 80 - 100.
- <span style="color:red;">Platelets: Unit: 1000/µL. Reference range: 150 - 450.
- <span style="color:red;">White Blood Cell Count (WBC): Unit: 1000/µL. Reference range: 4 - 11.
- <span style="color:red;">Neutrophils: Unit: %. Reference range: 55 - 70.
- <span style="color:red;">Band Neutrophils: Unit: %. Reference range: 0 - 6.
- <span style="color:red;">Lymphocytes: Unit: %. Reference range: 20 - 40.
- <span style="color:red;">C-Reactive Protein (CRP): Unit: mg/L. Reference range: 0 - 10.
- <span style="color:red;">Methemoglobin: Unit: %. Reference range: 0 - 2.
- <span style="color:red;">Creatine Kinase (CK): Unit: U/L. Reference range: 30 - 200.
- <span style="color:red;">Creatine Kinase-MB (CK-MB): Unit: ng/mL. Reference range: 0 - 5.
- <span style="color:red;">Troponin T: Unit: ng/mL. Reference range: 0 - 14.
- Height: Unit: cm. Reference range: /.
- Weight: Unit: kg. Reference range: /.

Here is an example of input information:  
Example #1:  
Input information of a patient:  
The patient is a female, aged 52 years. 
The patient <span style="color:red;">has data from 6 hours</span> that occurred at 0, 1, 2, 3<span style="color:red;">, 4, 5</span>.  
Details of the features for each visit are as follows:
- Heart Rate: “73, 77, 86, 81, 95, 92”
- …

RESPONSE:  
<span style="color:red;">{
    "diagnosis": label_text,
    "probability": "an integer (0 to 100) indicating how likely the risk of mortality/aki/sepsis",
    "explanation": "a brief explanation for the prediction",
}</span>

Input information of a patient:  
The patient is a male, aged 50.0 years. 
The patient <span style="color:red;">has data from 6 hours</span> that occurred at 0, 1, 2, 3<span style="color:red;">, 4, 5</span>.  
Details of the features for each visit are as follows:
- Heart Rate: "75.34, 84.21, 92.45, 68.72<span style="color:red;">, 87.15, 78.93</span>"
- Systolic Blood Pressure: "112.24, 105.78, 118.36, 94.52<span style="color:red;">, 110.87, 115.19</span>"
- Diastolic Blood Pressure: "72.41, 65.29, 78.56, 74.83<span style="color:red;">, 68.12, 76.97</span>"
- Mean <span style="color:red;">Arterial</span> Pressure <span style="color:red;">(MAP)</span>: "85.68, 78.45, 92.13, 72.87<span style="color:red;">, 82.35, 89.54</span>"
- Oxygen Saturation: "98.32, 96.45, 97.78, 95.23<span style="color:red;">, 99.07, 97.64</span>"
- Respiratory Rate: "16.25, 14.78, 18.32, 13.46<span style="color:red;">, 19.58, 15.91</span>"
- Temperature: "36.82, 37.14, 36.63, 37.35<span style="color:red;">, 36.98, 37.05</span>"
- pH <span style="color:red;">Level</span>: "7.38, 7.41, 7.36, 7.43<span style="color:red;">, 7.40, 7.37</span>"
- <span style="color:red;">Partial Pressure of Oxygen (PaO2): "88.76, 92.35, 85.43, 98.67, 78.29, 94.14"
- <span style="color:red;">Partial Pressure of Carbon Dioxide (PaCO2): "38.42, 42.17, 36.85, 40.29, 44.53, 39.76"
- <span style="color:red;">Base Excess: "0.24, -1.38, 1.45, -2.17, 2.06, 0.58"
- <span style="color:red;">Bicarbonate: "25.36, 23.48, 27.29, 24.75, 26.34, 28.17"
- Fraction of Inspired Oxygen <span style="color:red;">(FiO2)</span>: "21.00, 24.35, 28.72, 21.00<span style="color:red;">, 35.45, 30.18</span>"
- <span style="color:red;">International Normalized Ratio (INR): "0.92, 1.03, 1.15, 0.95, 1.18, 1.07"
- <span style="color:red;">Partial Thromboplastin Time (PTT): "28.46, 32.19, 27.85, 30.42, 33.67, 29.38"
- <span style="color:red;">Fibrinogen: "310.27, 275.64, 350.42, 240.93, 380.51, 290.75"
- <span style="color:red;">Sodium: "138.47, 142.23, 136.85, 140.32, 139.58, 143.15"
- <span style="color:red;">Potassium: "3.94, 4.27, 3.73, 4.56, 4.08, 4.82"
- <span style="color:red;">Chloride: "101.37, 98.45, 103.28, 100.56, 105.18, 97.82"
- <span style="color:red;">Calcium: "9.24, 9.82, 8.94, 10.12, 9.53, 9.07"
- <span style="color:red;">Ionized Calcium: "1.22, 1.15, 1.27, 1.19, 1.24, 1.28"
- <span style="color:red;">Magnesium: "1.93, 2.12, 1.84, 2.05, 1.88, 2.17"
- <span style="color:red;">Phosphate: "3.24, 3.85, 2.93, 4.06, 3.52, 3.07"
- Glucose: "95.42, 110.87, 85.29, 120.63<span style="color:red;">, 105.48, 92.75</span>"
- <span style="color:red;">Lactate: "1.23, 0.85, 1.54, 1.08, 1.82, 0.73"
- <span style="color:red;">Albumin: "4.23, 3.87, 4.56, 3.62, 4.05, 4.34"
- <span style="color:red;">Alkaline Phosphatase: "78.56, 95.32, 120.75, 60.43, 135.28, 85.96"
- <span style="color:red;">Alanine Aminotransferase (ALT): "25.47, 18.35, 35.82, 15.63, 42.18, 29.54"
- <span style="color:red;">Aspartate Aminotransferase (AST): "22.34, 18.76, 32.45, 15.27, 38.93, 25.18"
- <span style="color:red;">Total Bilirubin: "0.54, 0.82, 0.37, 1.05, 0.73, 0.45"
- <span style="color:red;">Direct Bilirubin: "0.12, 0.06, 0.17, 0.21, 0.09, 0.14"
- <span style="color:red;">Blood Urea Nitrogen (BUN): "15.32, 10.78, 18.45, 8.96, 16.23, 12.57"
- <span style="color:red;">Creatinine: "0.93, 1.14, 0.73, 1.25, 0.86, 1.07"
- <span style="color:red;">Urine Output: "45.36, 38.87, 42.53, 35.29, 48.64, 40.12"
- <span style="color:red;">Hemoglobin: "14.53, 13.82, 15.24, 13.27, 14.86, 16.08"
- <span style="color:red;">Mean Corpuscular Hemoglobin (MCH): "29.45, 31.27, 28.63, 32.15, 30.56, 27.84"
- <span style="color:red;">Mean Corpuscular Hemoglobin Concentration (MCHC): "33.35, 34.22, 35.08, 32.47, 34.75, 33.63"
- <span style="color:red;">Mean Corpuscular Volume (MCV): "88.24, 92.75, 85.36, 95.48, 90.32, 87.69"
- <span style="color:red;">Platelets: "230.46, 310.85, 180.27, 270.53, 350.92, 210.39"
- <span style="color:red;">White Blood Cell Count (WBC): "7.53, 6.28, 8.57, 5.56, 9.24, 6.83"
- <span style="color:red;">Neutrophils: "60.47, 65.23, 58.84, 67.35, 62.18, 59.76"
- <span style="color:red;">Band Neutrophils: "2.34, 3.45, 1.27, 4.18, 2.56, 3.09"
- <span style="color:red;">Lymphocytes: "32.45, 25.82, 35.47, 28.63, 30.92, 33.18"
- <span style="color:red;">C-Reactive Protein (CRP): "3.56, 1.83, 5.24, 2.53, 8.07, 4.28"
- <span style="color:red;">Methemoglobin: "0.52, 1.05, 0.37, 1.58, 0.84, 0.43"
- <span style="color:red;">Creatine Kinase (CK): "95.36, 120.47, 80.28, 150.93, 110.52, 65.79"
- <span style="color:red;">Creatine Kinase-MB (CK-MB): "2.53, 1.86, 3.24, 0.95, 2.87, 1.58"
- <span style="color:red;">Troponin T: "6.24, 3.17, 8.45, 2.09, 9.68, 5.32"
- Height: "183.00, 183.00, 183.00, 183.00<span style="color:red;">, 183.00, 183.00</span>"
- Weight: "78.50, 78.50, 78.50, 78.50<span style="color:red;">, 78.50, 78.50</span>"

Please follow the Chain-of-Thought Analysis Process:
1. Analyze the data step by step, For example:
    - Blood pressure shows a slight downward trend, indicating...
    - Heart rate is stable, suggesting...
    - Lab results indicate [specific condition or lack thereof]...
    - The patient underwent [specific intervention], which could mean...

2. Make Intermediate Conclusions:
    - Draw intermediate conclusions from each piece of data. For example:
        - If a patient’s blood pressure is consistently low, it might indicate poor cardiovascular function.
        - The patient’s cardiovascular function is [conclusion].
        - [Other intermediate conclusions based on data].

3. Aggregate the Findings:
    - After analyzing each piece of data, aggregate these findings to form a comprehensive view of the patient’s condition.
    - Summarize key points from the initial analysis and intermediate conclusions.

    Aggregated Findings:
    - Considering the patient’s vital signs and lab results, the overall health status is...

4. Final Assessment:
    - Conclude with an assessment of the likelihood of the patient not surviving their hospital stay<span style="color:red;">/having acute kidney injury/sepsis at the end of the data batch</span>.
    - <span style="color:red;">Follow the instructions to provide output.</span> 

Example Chain-of-Thought Analysis:

1. Analyze the data step by step:
    - Blood pressure shows a slight downward trend, which might indicate a gradual decline in cardiovascular stability.
    - Heart rate is stable, which is a good sign, suggesting no immediate cardiac distress.
    - Elevated white blood cell count could indicate an infection or an inflammatory process in the body.
    - Low potassium levels might affect heart rhythm and overall muscle function.

2. Make Intermediate Conclusions:
    - The decreasing blood pressure could be a sign of worsening heart function or infection-related hypotension.
    - Stable heart rate is reassuring but does not completely rule out underlying issues.
    - Possible infection, considering the elevated white blood cell count.
    - Potassium levels need to be corrected to prevent cardiac complications.

3. Aggregate the Findings:
    - The patient is possibly facing a cardiovascular challenge, compounded by an infection and electrolyte imbalance.

    Aggregated Findings:
    - Considering the downward trend in blood pressure, stable heart rate, signs of infection, and electrolyte imbalance, the patient’s overall health status seems to be moderately compromised.

4. Final Assessment:  

    <span style="color:red;">{'diagnosis': 'diagnosis' or 'not-diagnosis', 'probability': 65, 'explanation': 'Moderately compromised condition due to decreasing blood pressure, stable heart rate, signs of infection and electrolyte imbalance.'}</span>

RESPONSE:
