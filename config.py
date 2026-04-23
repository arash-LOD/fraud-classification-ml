"""
config.py
---------
Central configuration for the Fraud Classification ML project.
All paths, hyperparameter defaults, and training settings live here.
"""

import os


class Config:
    # ── Paths ────────────────────────────────────────────────────────────────
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR    = os.path.join(BASE_DIR, "data")
    MODELS_DIR  = os.path.join(BASE_DIR, "outputs", "models")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    PLOTS_DIR   = os.path.join(BASE_DIR, "outputs", "plots")
    LOGS_DIR    = os.path.join(BASE_DIR, "outputs", "logs")

    # Default dataset (Kaggle Credit Card Fraud Detection)
    DEFAULT_DATA = os.path.join(DATA_DIR, "creditcard.csv")

    # ── Data ─────────────────────────────────────────────────────────────────
    TARGET_COL    = "Class"   # 0 = legitimate, 1 = fraud
    TEST_SIZE     = 0.20
    RANDOM_STATE  = 42
    APPLY_SMOTE   = True

    # ── Training ─────────────────────────────────────────────────────────────
    CV_FOLDS = 5
    SCORING  = "f1"           # metric used for cross-validation and tuning

    # ── Model defaults ────────────────────────────────────────────────────────
    DEFAULT_MODEL  = "random_forest"

    # Logistic Regression
    LR_C          = 1.0
    LR_MAX_ITER   = 1000

    # Random Forest
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH    = 10
    RF_N_JOBS       = -1

    # XGBoost
    XGB_N_ESTIMATORS    = 200
    XGB_MAX_DEPTH       = 6
    XGB_LEARNING_RATE   = 0.05
    XGB_SCALE_POS_WEIGHT = 100   # roughly (# negatives / # positives)

    # ── Output ────────────────────────────────────────────────────────────────
    PREDICTION_OUTPUT = os.path.join(OUTPUTS_DIR, "predictions.csv")

    @classmethod
    def make_dirs(cls):
        """Create all output directories if they don't exist."""
        for d in [cls.DATA_DIR, cls.MODELS_DIR, cls.OUTPUTS_DIR, cls.PLOTS_DIR, cls.LOGS_DIR]:
            os.makedirs(d, exist_ok=True)
