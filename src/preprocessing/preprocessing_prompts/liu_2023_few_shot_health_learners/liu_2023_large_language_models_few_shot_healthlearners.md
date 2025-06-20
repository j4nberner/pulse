## Large Language Models are Few-Shot Health Learners

Paper: https://arxiv.org/pdf/2305.15525

Approach:

- Featurename unit: [Raw Data]
- Few shot examples are marked as Example Question with the answer

---

### Prompt used in paper:

Question:
Classify the given interbeat interval sequence in ms as either Atrial Fibrillation or Normal.
Sinus: 896,1192,592,1024,1072,808,888,896,760,1000,784,736,856,1000,1272,824,
872,1120,840,896,888,560,1248,824,968,960,1000,1008,776,744,896,1256.

Answer: "

### Adapted prompt:

Example Question: Classify the following ICU patient data as either aki or not-aki

age: 80.0
sex: Male
Height cm: 170.0
Weight kg: 60.0
Albumin g/dL: [2.25, 2.25, 2.25, 2.25, 2.25, 2.25]
Alkaline Phosphatase U/L: [104.3, 104.3, 104.3, 104.3, 104.3, 104.3]
...

Answer:
{
"diagnosis": "not-aki",
"probability": "the probability of your estimation as a float",
"explanation": "a brief explanation for the prediction"
}

Question: Classify the following ICU patient data as either aki or not-aki

age: 55.0
sex: Male
Height cm: 170.0
Weight kg: 95.0
Albumin g/dL: [2.25, 2.25, 2.25, 2.25, 2.25, 2.25]
Alkaline Phosphatase U/L: [104.3, 104.3, 104.3, 104.3, 104.3, 104.3]
Alanine Aminotransferase (ALT) U/L: [151.02, 24.0, 24.0, 24.0, 24.0, 24.0]
Aspartate Aminotransferase (AST) U/L: [231.81, 25.0, 25.0, 25.0, 25.0, 25.0]
Base Excess mmol/L: [-0.44, -1.0, -1.0, -1.0, -1.2, -1.2]
...
