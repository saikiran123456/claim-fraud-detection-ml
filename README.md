# Vehicle Insurance Claim Fraud Detection

## 1. Problem statement

Vehicle insurance claim fraud is a classification problem in which historical
claim, policy, vehicle and claimant information is used to distinguish between
potentially fraudulent and non-fraudulent claims.

The target variable is `FraudFound_P`:

- `0` – Non-fraudulent claim
- `1` – Fraudulent claim

The objective is to compare five machine learning classifiers and evaluate
their ability to identify potentially fraudulent insurance claims.
Predictions are intended to support further human investigation and do not
establish fraud on their own.

## 2. Dataset description

- **Source:** [Kaggle – Vehicle Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
- **Dataset file:** `data/fraud_oracle.csv`
- **Total records:** 15,420
- **Total columns including target:** 33
- **Predictor features used:** 31
- **Target variable:** `FraudFound_P`
- **Non-fraudulent records:** 14,497
- **Fraudulent records:** 923
- **Training records:** 12,336
- **Testing records:** 3,084
- **Train-test approach:** Stratified train-test split
- **Random state:** 42

The dataset represents historical vehicle insurance claims.

## 3. GitHub repository

[Project source code and files](https://github.com/saikiran123456/claim-fraud-detection-ml)

## 4. Streamlit live application deployed:

[Claim Fraud Detection ML](https://claim-fraud-detection-ml.streamlit.app/)

## 5. Models used and evaluation metrics

All five classifiers were evaluated using the same training and testing
partitions. The fraud class (`1`) was treated as the positive class when
calculating Precision, Recall, F1 and MCC. AUC represents binary ROC-AUC.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6436 | 0.8063 | 0.1320 | 0.8865 | 0.2299 | 0.2492 |
| Decision Tree (Entropy) | 0.6540 | 0.7979 | 0.1355 | 0.8865 | 0.2351 | 0.2557 |
| KNN | 0.9248 | 0.5948 | 0.1385 | 0.0486 | 0.0720 | 0.0485 |
| Gaussian Naive Bayes | 0.0973 | 0.5387 | 0.0608 | 0.9730 | 0.1145 | 0.0173 |
| SVM | 0.7866 | 0.8290 | 0.1738 | 0.6811 | 0.2769 | 0.2657 |


### Model-wise observations

- **Logistic Regression:** Accuracy is close to the majority-class reference, so the fraud-class metrics are particularly important. The model identifies a substantial share of actual fraud cases, but its low precision indicates a high number of false alerts. ROC-AUC = 0.8063, F1 = 0.2299, MCC = 0.2492.
- **Decision Tree (Entropy):** Accuracy is close to the majority-class reference, so the fraud-class metrics are particularly important. The model identifies a substantial share of actual fraud cases, but its low precision indicates a high number of false alerts. ROC-AUC = 0.7979, F1 = 0.2351, MCC = 0.2557.
- **KNN:** Accuracy is close to the majority-class reference, so the fraud-class metrics are particularly important. The precision-recall results indicate scope for improvement in identifying fraudulent claims. ROC-AUC = 0.5948, F1 = 0.0720, MCC = 0.0485.
- **Gaussian Naive Bayes:** Accuracy is close to the majority-class reference, so the fraud-class metrics are particularly important. The model identifies a substantial share of actual fraud cases, but its low precision indicates a high number of false alerts. ROC-AUC = 0.5387, F1 = 0.1145, MCC = 0.0173.
- **SVM:** Accuracy is close to the majority-class reference, so the fraud-class metrics are particularly important. The model identifies a substantial share of actual fraud cases, but its low precision indicates a high number of false alerts. ROC-AUC = 0.8290, F1 = 0.2769, MCC = 0.2657.

### Overall winner for the dataset

**SVM**

The model achieved the highest F1-score under the selected ranking criteria,
with an F1-score of 0.2769, MCC of 0.2657
and ROC-AUC of 0.8290.

The individual metric leaders were:

- Highest Accuracy: KNN (0.9248)
- Highest AUC: SVM (0.8290)
- Highest Precision: SVM (0.1738)
- Highest Recall: Gaussian Naive Bayes (0.9730)
- Highest F1: SVM (0.2769)
- Highest MCC: SVM (0.2657)

The overall ranking is based on the current held-out test results.

## 6. Project files

- `notebooks/fraud_detection.ipynb` – Data analysis, preprocessing,
  model development and evaluation.
- `app.py` – Streamlit application for model-based claim assessment.
- `requirements.txt` – Python package dependencies.
- `environment.yml` – Environment specification.
- `data/fraud_oracle.csv` – Original dataset.
- `data/test_data.csv` – Labelled test dataset.
- `model/` – Saved trained model pipelines.
- `models/` – Model copies used by the Streamlit application.
- `outputs/model_comparison.csv` – Model comparison results.
- `outputs/run_metadata.json` – Experiment configuration and results.
