"""
data_preprocessing.py
Handles loading, cleaning, and feature engineering for fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_data(filepath):
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset shape: {df.shape}")
    return df


def inspect_data(df):
    print("\n=== Dataset Info ===")
    print(df.info())
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    if "Class" in df.columns:
        print(df["Class"].value_counts())
        print(f"Fraud rate: {df['Class'].mean() * 100:.4f}%")


def clean_data(df):
    original_size = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    logger.info(f"Removed {original_size - len(df)} rows. Remaining: {len(df)}")
    return df


def feature_engineering(df):
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24
    if "Amount" in df.columns:
        df["Log_Amount"] = np.log1p(df["Amount"])
        df = df.drop(columns=["Amount", "Time"], errors="ignore")
    return df


def encode_categoricals(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        if col != "Class":
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def split_and_scale(df, target_col="Class", test_size=0.2, random_state=42, apply_smote=True):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, scaler


def preprocess_pipeline(filepath, target_col="Class", test_size=0.2, random_state=42, apply_smote=True):
    df = load_data(filepath)
    inspect_data(df)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_categoricals(df)
    return split_and_scale(df, target_col, test_size, random_state, apply_smote)
