"""
model.py
Defines, trains, and persists ML models for fraud classification.
Supports: Logistic Regression, Random Forest, XGBoost, and a Voting Ensemble.
"""

import os
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42
    ),
    "xgboost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=100, eval_metric="logloss",
        use_label_encoder=False, random_state=42
    ),
}

PARAM_GRIDS = {
    "logistic_regression": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "saga"]},
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 10, None],
        "min_samples_split": [2, 5],
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
    },
}


def get_model(name):
    """Return a model instance by name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODELS.keys())}")
    return MODELS[name]


def train_model(model, X_train, y_train):
    """Fit a model on training data."""
    logger.info(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def tune_model(name, X_train, y_train, cv=5, scoring="f1"):
    """Run GridSearchCV for hyperparameter tuning."""
    if name not in PARAM_GRIDS:
        raise ValueError(f"No param grid for '{name}'.")
    logger.info(f"Tuning {name} with GridSearchCV...")
    base = get_model(name)
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=base, param_grid=PARAM_GRIDS[name],
        cv=cv_strategy, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    logger.info(f"Best params: {grid.best_params_}  Best {scoring}: {grid.best_score_:.4f}")
    return grid.best_estimator_


def build_ensemble(X_train, y_train):
    """Build a soft-voting ensemble of all three base models."""
    logger.info("Building voting ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ("lr", get_model("logistic_regression")),
            ("rf", get_model("random_forest")),
            ("xgb", get_model("xgboost")),
        ],
        voting="soft", n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    logger.info("Ensemble training complete.")
    return ensemble


def save_model(model, path):
    """Persist a trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path):
    """Load a trained model from disk."""
    logger.info(f"Loading model from {path}")
    return joblib.load(path)


def predict(model, X):
    """Return class predictions."""
    return model.predict(X)


def predict_proba(model, X):
    """Return fraud probability estimates."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise AttributeError(f"{model.__class__.__name__} does not support predict_proba.")
