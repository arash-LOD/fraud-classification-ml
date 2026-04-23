"""
main.py - Entry point for the Fraud Classification ML pipeline.

Usage:
  python main.py --data data/creditcard.csv --mode train
  python main.py --data data/creditcard.csv --mode tune --model xgboost
  python main.py --data data/creditcard.csv --mode ensemble
  python main.py --data data/new_data.csv --mode predict --model-path outputs/models/xgboost.pkl
"""

import argparse
import logging
import os

from src.data_preprocessing import preprocess_pipeline
from src.model import (
    get_model, train_model, tune_model, build_ensemble,
    save_model, load_model, predict, predict_proba
)
from src.evaluate import evaluate_all
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fraud Classification ML Pipeline")
    parser.add_argument("--data",       type=str, required=True)
    parser.add_argument("--mode",       type=str, default="train",
                        choices=["train", "tune", "ensemble", "predict"])
    parser.add_argument("--model",      type=str, default="random_forest",
                        choices=["logistic_regression", "random_forest", "xgboost"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--no-smote",   action="store_true")
    return parser.parse_args()


def run_train(args):
    logger.info(f"=== TRAIN | model={args.model} ===")
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        args.data, Config.TARGET_COL, Config.TEST_SIZE, Config.RANDOM_STATE, not args.no_smote
    )
    model = train_model(get_model(args.model), X_train, y_train)
    metrics = evaluate_all(y_test, predict(model, X_test), predict_proba(model, X_test), args.model)
    save_model(model,  os.path.join(Config.MODELS_DIR, f"{args.model}.pkl"))
    save_model(scaler, os.path.join(Config.MODELS_DIR, "scaler.pkl"))
    logger.info(metrics)


def run_tune(args):
    logger.info(f"=== TUNE | model={args.model} ===")
    X_train, X_test, y_train, y_test, _ = preprocess_pipeline(
        args.data, Config.TARGET_COL, Config.TEST_SIZE, Config.RANDOM_STATE, not args.no_smote
    )
    best = tune_model(args.model, X_train, y_train, Config.CV_FOLDS, Config.SCORING)
    metrics = evaluate_all(y_test, predict(best, X_test), predict_proba(best, X_test), f"{args.model}_tuned")
    save_model(best, os.path.join(Config.MODELS_DIR, f"{args.model}_tuned.pkl"))
    logger.info(metrics)


def run_ensemble(args):
    logger.info("=== ENSEMBLE ===")
    X_train, X_test, y_train, y_test, _ = preprocess_pipeline(
        args.data, Config.TARGET_COL, Config.TEST_SIZE, Config.RANDOM_STATE, not args.no_smote
    )
    ens = build_ensemble(X_train, y_train)
    metrics = evaluate_all(y_test, predict(ens, X_test), predict_proba(ens, X_test), "ensemble")
    save_model(ens, os.path.join(Config.MODELS_DIR, "ensemble.pkl"))
    logger.info(metrics)


def run_predict(args):
    if not args.model_path:
        raise ValueError("--model-path required for predict mode.")
    import pandas as pd
    df = pd.read_csv(args.data)
    scaler = load_model(os.path.join(Config.MODELS_DIR, "scaler.pkl"))
    X = scaler.transform(df)
    model = load_model(args.model_path)
    df["prediction"]  = predict(model, X)
    df["fraud_proba"] = predict_proba(model, X)
    out = os.path.join(Config.OUTPUTS_DIR, "predictions.csv")
    os.makedirs(Config.OUTPUTS_DIR, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Predictions saved to {out}")


def main():
    args = parse_args()
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    modes = {"train": run_train, "tune": run_tune, "ensemble": run_ensemble, "predict": run_predict}
    modes[args.mode](args)


if __name__ == "__main__":
    main()
