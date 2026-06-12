# AI-Powered Phishing Email Detector

A machine learning-based web application that detects whether an email is legitimate or a phishing attempt. The system uses Natural Language Processing (NLP) techniques and a Linear Support Vector Machine (Linear SVM) model trained on the Enron email dataset.

## Features

* Detects phishing and legitimate emails
* Web-based interface built with Streamlit
* Text preprocessing and TF-IDF feature extraction
* Comparison of multiple machine learning models
* Interactive model performance visualization
* Displays Accuracy, Precision, Recall, and F1-Score metrics

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Extraction using TF-IDF
4. Model Training
5. Model Comparison
6. Model Selection
7. Deployment with Streamlit

## Models Evaluated

The following machine learning algorithms were compared:

* Linear SVM
* Logistic Regression
* Naive Bayes
* Random Forest
* Decision Tree
* K-Nearest Neighbors (KNN)
* Gradient Boosting
* AdaBoost

## Best Model

After evaluating all models using Accuracy, Precision, Recall, and F1-Score, Linear SVM achieved the highest overall performance and was selected as the final deployment model.

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 98.86% |
| Precision | 98.44% |
| Recall    | 99.14% |
| F1 Score  | 98.79% |

## Project Structure

```text
phishing-detector/
│
├── app.py
├── compare_models.py
├── train_best_model.py
├── phishing_model_new.pkl
├── model_comparison.csv
├── emails.csv
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/phishing-email-detector.git
cd phishing-email-detector
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* Joblib
* Plotly

## Dataset

This project uses the Enron Email Dataset containing both legitimate and phishing email samples for supervised machine learning classification.

## Future Improvements

* URL extraction and analysis
* Explainable AI (XAI) features
* Deep learning-based text classification
* Email attachment analysis
* Multi-class classification (Safe, Spam, Phishing)

## Author

Mihisara Koralage

Software Engineering Undergraduate
University of Sri Jayewardenepura
