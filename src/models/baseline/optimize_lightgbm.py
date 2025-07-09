import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

def objective(trial, X, y):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),
        "num_leaves"   : trial.suggest_int("num_leaves", 10, 30),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
        "max_depth"    : trial.suggest_int("max_depth", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 400),
        "reg_alpha"    : trial.suggest_float("reg_alpha", 5.0, 20.0),
        "reg_lambda"   : trial.suggest_float("reg_lambda", 5.0, 20.0),
        "class_weight" : "balanced",
        "random_state" : 42,
        "verbose"      : -1,
    }
    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    return scores.mean()

def run_optimization(X: pd.DataFrame, y: pd.Series, n_trials: int = 20):

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X, y), n_trials=n_trials)
    return study.best_params
