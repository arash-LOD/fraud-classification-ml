# Fraud Classification ML

A Python machine learning project for fraudulent transaction classification using scikit-learn, XGBoost, and imbalanced-learn.

## Overview

This project builds, tunes, and evaluates machine learning models to detect fraudulent financial transactions. It handles the inherent class imbalance of fraud datasets using SMOTE oversampling and provides a complete pipeline from raw data to model evaluation.

## Project Structure

```
fraud-classification-ml/
├── src/
│   ├── __init__.py              # Package exports
│   ├── data_preprocessing.py    # Data loading, cleaning, feature engineering, SMOTE
│   ├── model.py                 # Model definitions, training, tuning, ensemble
│   └── evaluate.py              # Metrics, plots (ROC, PR curve, confusion matrix)
├── tests/
│   └── test_pipeline.py         # Unit & integration tests (pytest)
├── data/                        # Place your CSV dataset here (gitignored)
├── outputs/
│   ├── models/                  # Saved .pkl model files
│   ├── plots/                   # Generated evaluation plots
│   └── predictions.csv          # Inference output
├── main.py                      # CLI entry point
├── config.py                    # Centralised configuration
├── requirements.txt
├── .gitignore
└── README.md
```

## Models

| Model | Notes |
|---|---|
| Logistic Regression | Fast baseline, balanced class weights |
| Random Forest | Ensemble tree model, handles non-linearity |
| XGBoost | Gradient boosting, `scale_pos_weight` for imbalance |
| Voting Ensemble | Soft-voting combination of all three models |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your dataset

Place your CSV file in the `data/` folder. This project is designed for the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), but any binary classification CSV works.

```
data/
└── creditcard.csv
```

### 3. Train a model

```bash
# Train with Random Forest (default)
python main.py --data data/creditcard.csv --mode train

# Train with XGBoost
python main.py --data data/creditcard.csv --mode train --model xgboost

# Train Logistic Regression without SMOTE
python main.py --data data/creditcard.csv --mode train --model logistic_regression --no-smote
```

### 4. Tune hyperparameters

```bash
python main.py --data data/creditcard.csv --mode tune --model xgboost
```

### 5. Train the voting ensemble

```bash
python main.py --data data/creditcard.csv --mode ensemble
```

### 6. Predict on new data

```bash
python main.py --data data/new_transactions.csv --mode predict --model-path outputs/models/xgboost.pkl
```

## Evaluation Outputs

After training, the following are saved to `outputs/`:

- **Confusion matrix** PNG
- **ROC curve** PNG
- **Precision-Recall curve** PNG
- **Trained model** .pkl file
- **Scaler** .pkl file

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Configuration

All settings (paths, hyperparameters, CV folds, scoring metric) can be adjusted in `config.py`.

## Dependencies

- Python 3.9+
- scikit-learn, xgboost, imbalanced-learn
- pandas, numpy, matplotlib, seaborn
- joblib (model persistence)

## License

MIT
