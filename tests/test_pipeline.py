"""
tests/test_pipeline.py
----------------------
Unit and integration tests for the Fraud Classification ML pipeline.
Run with:  pytest tests/ -v --cov=src
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.data_preprocessing import clean_data, feature_engineering, encode_categoricals, split_and_scale
from src.model import get_model, train_model, predict, predict_proba
from src.evaluate import compute_metrics


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a small synthetic fraud dataset."""
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=6,
        weights=[0.97, 0.03], random_state=42
    )
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 11)])
    df["Amount"] = np.abs(np.random.randn(500)) * 100
    df["Time"]   = np.arange(500) * 3600
    df["Class"]  = y
    return df


@pytest.fixture
def split_data(sample_df):
    """Return a train/test split from the sample dataset."""
    return split_and_scale(sample_df, target_col="Class", apply_smote=False)


# ── Data preprocessing tests ─────────────────────────────────────────────────

class TestDataPreprocessing:

    def test_clean_data_removes_duplicates(self, sample_df):
        df_dup = pd.concat([sample_df, sample_df.iloc[:10]], ignore_index=True)
        cleaned = clean_data(df_dup)
        assert len(cleaned) == len(sample_df)

    def test_feature_engineering_drops_columns(self, sample_df):
        df_eng = feature_engineering(sample_df.copy())
        assert "Amount" not in df_eng.columns
        assert "Time" not in df_eng.columns
        assert "Log_Amount" in df_eng.columns
        assert "Hour" in df_eng.columns

    def test_encode_categoricals_no_change_on_numeric(self, sample_df):
        df_num = sample_df.select_dtypes(include=[np.number])
        df_enc = encode_categoricals(df_num.copy())
        assert df_enc.shape == df_num.shape

    def test_split_and_scale_shapes(self, split_data):
        X_train, X_test, y_train, y_test, scaler = split_data
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test)  == X_test.shape[0]


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModels:

    @pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "xgboost"])
    def test_train_and_predict(self, model_name, split_data):
        X_train, X_test, y_train, y_test, _ = split_data
        model = get_model(model_name)
        model = train_model(model, X_train, y_train)
        preds = predict(model, X_test)
        assert preds.shape == y_test.shape
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "xgboost"])
    def test_predict_proba_range(self, model_name, split_data):
        X_train, X_test, y_train, _, _ = split_data
        model = get_model(model_name)
        model = train_model(model, X_train, y_train)
        probas = predict_proba(model, X_test)
        assert probas.min() >= 0.0
        assert probas.max() <= 1.0

    def test_get_model_invalid_name(self):
        with pytest.raises(ValueError):
            get_model("nonexistent_model")


# ── Evaluation tests ──────────────────────────────────────────────────────────

class TestEvaluation:

    def test_compute_metrics_keys(self, split_data):
        X_train, X_test, y_train, y_test, _ = split_data
        model = get_model("logistic_regression")
        model = train_model(model, X_train, y_train)
        y_pred  = predict(model, X_test)
        y_proba = predict_proba(model, X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            assert key in metrics

    def test_metrics_in_valid_range(self, split_data):
        X_train, X_test, y_train, y_test, _ = split_data
        model = get_model("logistic_regression")
        model = train_model(model, X_train, y_train)
        y_pred = predict(model, X_test)
        metrics = compute_metrics(y_test, y_pred)
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert 0.0 <= metrics[key] <= 1.0
