<p align="center">
  <img src="src/pulse_logo.png" alt="PULSE Logo" width="300"/>
</p>

# PULSE Benchmark

PULSE (_<u>P</u>redictive <u>U</u>nderstanding of <u>L</u>ife-threatening <u>S</u>ituations using <u>E</u>mbeddings_) benchmarks the predictive capabilities of Large Language Models (LLMs) using ICU time-series data.



## Overview

This repository contains the implementation for predicting mortality, acute kidney injury (AKI) and sepsis in intensive care unit (ICU) patients using Large Language Models (LLMs). Conventional machine learning and deep learning models serve as baseline comparisons to previously published LLM prompting and fine-tuning methods. Data preparation and model implementation is set up in a highly configurable manner to allow for flexible experimental design.

![Framework Overview](src/framework.png)

## Getting started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sepsis-prediction-llm.git
   cd sepsis-prediction-llm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
├── README.md
├── requirements.txt
├── config_train.yaml
├── config_benchmark.yaml
├── model_configs/
│   └── exampleModel.yaml
├── train_models.py
├── benchmark_models.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── modelmanager.py
│   │   └── example_model.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── preprocessing_advanced/
│   │   ├── preprocessing_baseline/
│   │   └── preprocessing_prompts/
|   └── framework.png
├── notebooks/
├── datasets/
└── secrets/
```

## Data

**Datasets:**

- HiRID (Switzerland, single site)
- MIMIC-IV (US, single-site)
- eICU (US, multi-site)

**Harmonization:**

Variable mapping, artifact removal, unit harmonization, cohort and variable selection was conducted according to the YAIB workflow (https://github.com/rvandewater/YAIB). Resulting cohorts vary between tasks with overlapping stay_ids within a dataset. 

## Tasks

Task Definitions are in accordance with YAIB (https://arxiv.org/abs/2306.05109).

1) **Mortality** 
   - **Task Description**: Binary classification, one prediction per stay, patient status at the end of the hospital stay predicted at 25h after ICU admission
   - **Data Structure**: Hourly data of first 25h of ICU stay for cases and controls, one label per stay_id

2) **Acute Kidney Injury (AKI)**
   - **Task Description**: Binary classification, multiple predictions per stay possible and dependent on data/prediction window setup
   - **Data Structure**: Hourly data and hourly labels, whole ICU stay duration for controls, until 12h after defined AKI onset for cases

3) **Sepsis**
   - **Task Description**: Binary classification, multiple predictions per stay possible and dependent on data/prediction window setup
   - **Data Structure**: Hourly data and hourly labels, whole ICU stay duration for controls, until 12h after defined sepsis onset for cases

## Implemented Models

| Type | Models                          |
| -------- | ------------------------------- |
| **conventional ML**   | RandomForest, XGBoost, LightGBM |
| **conventional DL**   | CNN, LSTM, GRU, InceptionTime   |
| **LLM**  | Llama 3.1-8b                    |

## Results

A Metric Tracker is running alongside each training / evaluation process. Predictions are tracked and evaluated of all validation and test runs and saved to a json file in the output.

## Train a model

1. adjust config_train.yaml
   - set debug flag to only load n rows of data (can be specified in dataloader.py)
   - set wandb flag and entity
   - set project base path
   - choose tasks & datasets to train and evaluate
   - choose models to train
2. run train_models.py

## Evaluate a model

1. adjust config_benchmark.yaml
2. run benchmark_models.py

## Add a new model

1. Add a new ExampleModel.yaml to model_configs/ with its config

   ```json
   - name: "ExampleModel"
      params:
         trainer_name: "ExampleTrainer"
         input_size: 784
         hidden_size: 128
         output_size: 10
   ```

2. List the model name in config_train.yaml under models

3. Add a new file in src/models which will host the model and the trainer class

   ```python
   class ExampleModel(PulseTemplateModel):
      def __init__(self, params: Dict[str, Any], **kwargs) -> None:
         super().__init__(model_name, trainer_name, params=params)
      def set_trainer(self, trainer_name, train_loader, val_loader, test_dataloader):
         self.trainer = ExampleTrainer(self,train_loader, val_loader, test_dataloader)
   ```

   ```python
   class ExampleTrainer():
      def __init__(self, model, train_loader, val_loader, test_loader):
         self.model = model
         self.train_loader = train_loader
         self.test_loader = test_loader


      def train(self):
         # training loop
         pass

      def validate(self, val_loader):
         # validation loop
         metrics_tracker = MetricsTracker()
   ```

4. add the new model name and import to the src/models/**init**.py
5. adjust **getitem** method in src/data/dataloader.py for model specific preprocessing
