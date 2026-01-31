# Voice Gender Recognition System

This project implements machine learning models to identify a voice as male or female based on acoustic properties of the voice and speech. It compares the performance of Decision Tree and Random Forest algorithms.

## Project Overview
Gender recognition by voice is a challenging problem in audio processing and biometric authentication. This system analyzes a dataset of acoustic features extracted from voice samples to perform binary classification (Male vs. Female).

## Dataset
The project uses the "Voice Gender" dataset, which consists of 3,168 recorded voice samples. Each sample is analyzed to extract acoustic properties, including:
* **Mean Frequency (meanfreq)**
* **Standard Deviation of Frequency (sd)**
* **Spectral Entropy (sp.ent)**
* **Fundamental Frequency (meanfun)**
* ...and other spectral features.

## Methodology
The following classification algorithms were implemented and compared:
1.  **Decision Tree Classifier:** A flow-chart-like structure where an internal node represents a feature, the branch represents a decision rule, and each leaf node represents the outcome.
2.  **Random Forest Classifier:** An ensemble learning method that operates by constructing a multitude of decision trees at training time to improve predictive accuracy and control over-fitting.

## Performance Analysis
Both models were evaluated using accuracy scores, F1 scores, and K-Fold Cross-Validation (K=10).

* **Random Forest:** Consistently achieved higher accuracy (typically ~97%), demonstrating better generalization and robustness against noise.
* **Decision Tree:** Achieved strong results (typically ~96%) but showed slightly more variance compared to the ensemble method.

## Technologies
* Python
* Pandas (Data Preprocessing)
* Scikit-Learn (Model Building & Evaluation)
* LabelEncoder (Target Transformation)
