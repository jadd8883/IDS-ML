
# Machine Learning-Based Intrusion Detection System (ML-IDS)

## Overview
A cybersecurity project that uses machine learning to classify
network traffic records as benign or potentially malicious.

The project uses a Random Forest classifier and a Streamlit
dashboard to display predictions and export analysis results.

## Features
- Network traffic data preprocessing
- Machine learning-based binary classification
- Random Forest model training
- Precision, recall, F1-score and confusion matrix evaluation
- Interactive Streamlit dashboard
- Suspicious traffic record filtering
- CSV export of prediction results

## Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn
- Joblib

## Project Structure
```text
ml-ids-project/
├── app.py
├── train.py
├── inspect_data.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
└── models/
```

## Dataset
Dataset: CIC-IDS2017

Official source:
https://www.unb.ca/cic/datasets/ids-2017.html

Download the dataset and place the selected CSV at:
data/network_traffic.csv

The CSV must contain a Label column and suitable numeric
network-flow features.

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset and place it in the data directory.
5. Train the model:

```bash
python train.py
```

6. Start the dashboard:

```bash
streamlit run app.py
```

## Evaluation
Evaluation metrics include accuracy, precision, recall,
F1 score and a confusion matrix.

Add your actual evaluation results after running the model.

## Limitations
- Predictions depend on the training dataset and selected features.
- CSV-based analysis is not live network monitoring.
- A flagged record is not proof of malicious activity.
- Results on benchmark data may not generalize to real networks.

## Future Improvements
- Multiclass attack classification
- Improved testing on independent traffic captures
- Alert logging and filtering
- Real-time flow monitoring
- Model comparison and explainability

## Disclaimer
For educational and authorized cybersecurity research.
