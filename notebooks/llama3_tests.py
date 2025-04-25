import time
from transformers import AutoTokenizer, AutoModelForCausalLM



# Sample input string for classification
# input_text = ("Below are Question-Answer pair examples of ICU data classified as sepsis or not-sepsis:\n"
# "Q: Classify the given ICU data sequence as either sepsis or not-sepsis:\n"
# "   Features:\n"
# "stay_id 1.0, age 0.7738563344194752, sex -1.0, Height [cm] 1.024295936172697, Weight [kg] 0.8846886959101604, Albumin [g/dL] 0.0, Alkaline Phosphatase [U/L] 0.0, Alanine Aminotransferase\n"
# "(ALT) [U/L] -0.0, Aspartate Aminotransferase\n"
# "(AST) [U/L] 0.0, Base Excess [mmol/L] -1.037447851818147, Bicarbonate [mmol/L] -1.0843500175366092, Total Bilirubin [mg/dL] -0.0, Direct Bilirubin [mg/dL] -0.0, Band Neutrophils [%] -0.0, Blood Urea Nitrogen\n"
# "(BUN) [mg/dL] -0.0, Calcium [mg/dL] 0.0, Ionized Calcium [mmol/L] 1.282673935399477, Creatine Kinase\n"
# "(CK) [U/L] 0.0, Creatine Kinase-MB\n"
# "(CK-MB) [ng/mL] 0.0, Chloride [mmol/L] 0.22903153720133215, Creatinine [mg/dL] 0.0, C-Reactive Protein\n"
# "(CRP) [mg/L] 0.0, Diastolic Blood Pressure\n"
# "[mmHg] -0.7487564081264964, Fibrinogen [mg/dL] -0.0, Fraction of Inspired Oxygen\n"
# "(FiO2) [%] -0.37401614227395164, Glucose [mg/dL] 0.22910249704153643, Hemoglobin [g/dL] -0.4123375851609974, Heart Rate [bpm] 0.4823897795527653, International Normalised Ratio\n"
# "(INR) [-] -0.4397102531971466, Potassium [mmol/L] 0.8181373819042236, Lactate [mmol/L] -0.0018181386934984208, Lymphocytes [%] -0.0, Mean Arterial Pressure (MAP) [mmHg] -0.988176941156571, Mean Corpuscular\n"
# "Hemoglobin (MCH) [pg] -0.3481111006563849, Mean Corpuscular Hemoglobin\n"
# "Concentration (MCHC) [g/dL] 0.16063690516035076, Mean Corpuscular\n"
# "Volume (MCV) [fL] -0.504585379513309, Methemoglobin [%] 1.1147597851781317, Magnesium [mg/dL] -0.0, Sodium [mmol/L] 0.2316891655355679, Neutrophils [%] -0.0, Oxygen Saturation [%] 1.0910523734414423, Partial Pressure of\n"
# "Carbon Dioxide (PaCO2) [mmHg] 0.1707487943934932, pH Level [-] -1.097254110837404, Phosphate [mg/dL] 0.0, Platelets [10^3/µL] -0.35233584979073596, Partial Pressure of\n"
# "Oxygen (PaO2) [mmHg] 1.1139767488081462, Partial Thromboplastin Time\n"
# "(PTT) [sec] 0.0, Respiratory Rate\n"
# "[breaths/min] -0.17111169152400021, Systolic Blood Pressure\n"
# "[mmHg] -1.0706591245391182, Temperature [°C] 0.0, Troponin T [ng/mL] -0.0, Urine Output [mL/h] -0.14642558424999422, White Blood Cell Count\n"
# "(WBC) [10^3/µL] -0.9362689800957276\n"
# "A: not-sepsis\n\n"
# "Q: Classify the given ICU data sequence as either sepsis or not-sepsis:\n"
# "   Features:\n"
# "stay_id 2.0, age 1.1031541976953103, sex -1.0, Height [cm] -0.1328817716204434, Weight [kg] -1.1015010120628486, Albumin [g/dL] 0.0, Alkaline Phosphatase [U/L] 0.0, Alanine Aminotransferase\n"
# "(ALT) [U/L] -0.2710885734848575, Aspartate Aminotransferase\n"
# "(AST) [U/L] -0.25832382536105714, Base Excess [mmol/L] 0.0, Bicarbonate [mmol/L] -0.0, Total Bilirubin [mg/dL] 0.3152154432866186, Direct Bilirubin [mg/dL] -0.22909450363216755, Band Neutrophils [%] -0.0, Blood Urea Nitrogen\n"
# "(BUN) [mg/dL] 0.353765149709523, Calcium [mg/dL] -0.04305519051415102, Ionized Calcium [mmol/L] -0.0, Creatine Kinase\n"
# "(CK) [U/L] 0.1973918224468484, Creatine Kinase-MB\n"
# "(CK-MB) [ng/mL] -0.27989379618764354, Chloride [mmol/L] -0.7524982197378528, Creatinine [mg/dL] -0.22939636878169137, C-Reactive Protein\n"
# "(CRP) [mg/L] -0.6359442365459125, Diastolic Blood Pressure\n"
# "[mmHg] -0.0, Fibrinogen [mg/dL] -0.0, Fraction of Inspired Oxygen\n"
# "(FiO2) [%] 0.0, Glucose [mg/dL] 0.08863579588194521, Hemoglobin [g/dL] 1.3886502189330037, Heart Rate [bpm] -1.4654064694384747, International Normalised Ratio\n"
# "(INR) [-] -0.5146446408880857, Potassium [mmol/L] 0.6395221756533823, Lactate [mmol/L] -0.0, Lymphocytes [%] -0.0, Mean Arterial Pressure (MAP) [mmHg] 1.4412937349103738, Mean Corpuscular\n"
# "Hemoglobin (MCH) [pg] 0.1360996412276623, Mean Corpuscular Hemoglobin\n"
# "Concentration (MCHC) [g/dL] 0.16063690516035076, Mean Corpuscular\n"
# "Volume (MCV) [fL] 0.2187789153665021, Methemoglobin [%] 0.0, Magnesium [mg/dL] 0.7206882691198644, Sodium [mmol/L] 0.6448040593844329, Neutrophils [%] -0.0, Oxygen Saturation [%] -0.6858600417007048, Partial Pressure of\n"
# "Carbon Dioxide (PaCO2) [mmHg] -0.0, pH Level [-] 0.0, Phosphate [mg/dL] 0.52701881897483, Platelets [10^3/µL] -0.4822401938191186, Partial Pressure of\n"
# "Oxygen (PaO2) [mmHg] 0.0, Partial Thromboplastin Time\n"
# "(PTT) [sec] 0.0, Respiratory Rate\n"
# "[breaths/min] 0.2346055850393312, Systolic Blood Pressure\n"
# "[mmHg] -0.0, Temperature [°C] -0.2744345955647103, Troponin T [ng/mL] -0.15486633275661604, Urine Output [mL/h] 0.5898578332615875, White Blood Cell Count\n"
# "(WBC) [10^3/µL] 0.122782797371873\n"
# "A: not-sepsis\n\n"
# "Q: Classify the given ICU data sequence as either sepsis or not-sepsis:\n"
# "   Features:\n"
# "stay_id 2.0, age 1.1031541976953103, sex -1.0, Height [cm] -0.1328817716204434, Weight [kg] -1.1015010120628486, Albumin [g/dL] 0.0, Alkaline Phosphatase [U/L] 0.0, Alanine Aminotransferase\n"
# "(ALT) [U/L] -0.0, Aspartate Aminotransferase\n"
# "(AST) [U/L] 0.0, Base Excess [mmol/L] 0.0, Bicarbonate [mmol/L] -0.0, Total Bilirubin [mg/dL] -0.0, Direct Bilirubin [mg/dL] -0.0, Band Neutrophils [%] -0.0, Blood Urea Nitrogen\n"
# "(BUN) [mg/dL] -0.0, Calcium [mg/dL] 0.0, Ionized Calcium [mmol/L] -0.0, Creatine Kinase\n"
# "(CK) [U/L] 0.0, Creatine Kinase-MB\n"
# "(CK-MB) [ng/mL] 0.0, Chloride [mmol/L] 0.0, Creatinine [mg/dL] 0.0, C-Reactive Protein\n"
# "(CRP) [mg/L] 0.0, Diastolic Blood Pressure\n"
# "[mmHg] -0.0, Fibrinogen [mg/dL] -0.0, Fraction of Inspired Oxygen\n"
# "(FiO2) [%] 0.0, Glucose [mg/dL] 0.4749192240708215, Hemoglobin [g/dL] -0.0, Heart Rate [bpm] -1.2856098926085142, International Normalised Ratio\n"
# "(INR) [-]: -0.0, Potassium [mmol/L] 0.0, Lactate [mmol/L] -0.0, Lymphocytes [%] -0.0, Mean Arterial Pressure (MAP) [mmHg] 0.12806634244175505, Mean Corpuscular\n"
# "Hemoglobin (MCH) [pg] 0.0, Mean Corpuscular Hemoglobin\n"
# "Concentration (MCHC) [g/dL] -0.0, Mean Corpuscular\n"
# "Volume (MCV) [fL] -0.0, Methemoglobin [%] 0.0, Magnesium [mg/dL] -0.0, Sodium [mmol/L] 0.0, Neutrophils [%] -0.0, Oxygen Saturation [%] -0.3304775586722754, Partial Pressure of\n"
# "Carbon Dioxide (PaCO2) [mmHg] -0.0, pH Level [-]: 0.0, Phosphate [mg/dL] 0.0, Platelets [10^3/µL] -0.0, Partial Pressure of\n"
# "Oxygen (PaO2) [mmHg] 0.0, Partial Thromboplastin Time\n"
# "(PTT) [sec] 0.0, Respiratory Rate\n"
# "[breaths/min] -0.19700853896421258, Systolic Blood Pressure\n"
# "[mmHg] -0.0, Temperature [°C] -0.2744345955647103, Troponin T [ng/mL] -0.0, Urine Output [mL/h] 0.23698542796292107, White Blood Cell Count\n"
# "(WBC) [10^3/µL] 0.0\n"
# "A: not-sepsis\n\n"
# "Q: Classify the given ICU data sequence as either sepsis or not-sepsis:\n"
# "   Features:\n"
# "stay_id: 30549.0, age: 1.4324520609711453, sex: 1.0, Height [cm]: -0.7114706255170136, Weight [kg]: 1.5467519319011633, Albumin [g/dL]: 0.0, Alkaline Phosphatase [U/L]: 0.0, Alanine Aminotransferase\n"
# "(ALT) [U/L]: -0.0, Aspartate Aminotransferase\n"
# "(AST) [U/L]: 0.0, Base Excess [mmol/L]: -0.9638033338839948, Bicarbonate [mmol/L]: -1.301928836295775, Total Bilirubin [mg/dL]: -0.0, Direct Bilirubin [mg/dL]: -0.0, Band Neutrophils [%]: -0.0, Blood Urea Nitrogen\n"
# "(BUN) [mg/dL]: -0.0, Calcium [mg/dL]: 0.0, Ionized Calcium [mmol/L]: 0.023806251746493024, Creatine Kinase\n"
# "(CK) [U/L]: 0.0, Creatine Kinase-MB\n"
# "(CK-MB) [ng/mL]: 0.0, Chloride [mmol/L]: 0.22903153720133215, Creatinine [mg/dL]: 0.0, C-Reactive Protein\n"
# "(CRP) [mg/L]: 0.0, Diastolic Blood Pressure [mmHg]: -0.6617079196521964, Fibrinogen [mg/dL]: -0.0, Fraction of Inspired Oxygen\n"
# "(FiO2) [%]: 0.0, Glucose [mg/dL]: -1.0702144886846816, Hemoglobin [g/dL]: 2.0772632028512974, Heart Rate [bpm]: 1.1116777984576274, International Normalised Ratio\n"
# "(INR) [-]: -0.0, Potassium [mmol/L]: -0.9680146806041949, Lactate [mmol/L]: -0.12554955538508422, Lymphocytes [%]: -0.0, Mean Arterial Pressure (MAP) [mmHg]: -0.988176941156571, Mean Corpuscular\n"
# "Hemoglobin (MCH) [pg]: 0.0, Mean Corpuscular Hemoglobin\n"
# "Concentration (MCHC) [g/dL]: -0.0, Mean Corpuscular\n"
# "Volume (MCV) [fL]: -0.0, Methemoglobin [%]: 2.5466448523337886, Magnesium [mg/dL]: -0.0, Sodium [mmol/L]: 0.025131718611135414, Neutrophils [%]: -0.0, Oxygen Saturation [%]: -1.3966250077575637, Partial Pressure of\n"
# "Carbon Dioxide (PaCO2) [mmHg]: -0.23610622824394534, pH Level [-]: -0.7393155960354231, Phosphate [mg/dL]: 0.0, Platelets [10^3/µL]: -0.0, Partial Pressure of\n"
# "Oxygen (PaO2) [mmHg]: -0.518704185425681, Partial Thromboplastin Time\n"
# "(PTT) [sec]: 0.0, Respiratory Rate\n"
# "[breaths/min]: 1.3568023074485451, Systolic Blood Pressure\n"
# "[mmHg]: -1.1520606169804606, Temperature [°C]: 0.0, Troponin T [ng/mL]: -0.0, Urine Output [mL/h]: -0.0, White Blood Cell Count\n"
# "(WBC) [10^3/µL]: 0.0\n"
# "A: Answer with yes or no.")

# Load the tokenizer and model from Hugging Face
model_name = "meta-llama/Llama-3.1-8B"  # Replace with the actual model name if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = (
    "You are an experienced doctor in the ICU. "
    "I will provide you with a sequence of ICU data, and you need to classify it as either sepsis or not-sepsis.\n"
    "Below are Question-Answer pair examples of ICU data classified as sepsis or not-sepsis:\n"
    "Q: Classify the given ICU data sequence as either sepsis or not-sepsis:\n"
    "Features:\n"
    "stay_id 1.0, age -0.5, sex 1.0, Height [cm] 0.1, Weight [kg] -0.2, Albumin [g/dL] 1.2, "
    "Alkaline Phosphatase [U/L] 0.3, Alanine Aminotransferase (ALT) [U/L] 0.2, Aspartate Aminotransferase (AST) [U/L] 0.1, "
    "Base Excess [mmol/L] 0.0, Bicarbonate [mmol/L] 0.1\n"
    "A: 0.1\n\n"
    "Your task is to classify the following ICU data sequence.\n"
    "Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of sepsis.\n"
    "Q: Classify the given ICU data sequence as either sepsis or not-sepsis:\n"
    "Features:\n"
    "stay_id 2.0, age 1.1031541976953103, sex -1.0, Height [cm] -0.1328817716204434, "
    "Weight [kg] -1.1015010120628486, Albumin [g/dL] 0.0, Alkaline Phosphatase [U/L] 0.0, "
    "Alanine Aminotransferase (ALT) [U/L] -0.2710885734848575, Aspartate Aminotransferase (AST) [U/L] "
    "-0.25832382536105714, Base Excess [mmol/L] 0.0, Bicarbonate [mmol/L] -0.0\n"
    "A:"
)



# Tokenize the input
start_time = time.time()
inputs = tokenizer(input_text, return_tensors="pt")
print("Tokenization time:", time.time() - start_time)

# Perform inference
start_time = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=5,
    do_sample=False,
    temperature=0.0,  # deterministic
)
print("Inference time:", time.time() - start_time)

# Decode and print the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Output:", output_text)

# Extract the answer (sepsis or not-sepsis) from the output
answer = output_text.split("A:")[-1].strip()
print("Answer:", answer)