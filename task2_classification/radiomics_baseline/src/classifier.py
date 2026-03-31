"""
Classifier factory and sklearn Pipeline builder for the radiomics baseline.

Supported classifiers
---------------------
  random_forest       sklearn RandomForestClassifier
  svm                 sklearn SVC  (probability=True forced — needed for AUROC)
  logistic_regression sklearn LogisticRegression
  xgboost             xgboost XGBClassifier  (optional; pip install xgboost)
  mlp                 sklearn MLPClassifier

The Pipeline is:  StandardScaler  →  Classifier
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def build_classifier(config):
    """
    Instantiate the classifier specified in config['classifier']['type'].

    Parameters
    ----------
    config : dict
        Full config dict loaded from YAML.

    Returns
    -------
    An sklearn-compatible estimator that exposes predict_proba().
    """
    clf_cfg     = config["classifier"]
    clf_type    = clf_cfg["type"]
    arch        = clf_cfg.get(clf_type, {})
    seed        = config.get("seed", 42)

    if clf_type == "random_forest":
        return RandomForestClassifier(
            n_estimators     = arch.get("n_estimators",     300),
            max_depth        = arch.get("max_depth",        None),
            min_samples_split = arch.get("min_samples_split", 2),
            min_samples_leaf  = arch.get("min_samples_leaf",  1),
            class_weight     = arch.get("class_weight",    "balanced"),
            n_jobs           = arch.get("n_jobs",           -1),
            random_state     = seed,
        )

    if clf_type == "svm":
        return SVC(
            C            = arch.get("C",            1.0),
            kernel       = arch.get("kernel",       "rbf"),
            gamma        = arch.get("gamma",        "scale"),
            class_weight = arch.get("class_weight", "balanced"),
            probability  = True,   # always on — needed for soft-vote + AUROC
            random_state = seed,
        )

    if clf_type == "logistic_regression":
        return LogisticRegression(
            C            = arch.get("C",            1.0),
            max_iter     = arch.get("max_iter",     2000),
            class_weight = arch.get("class_weight", "balanced"),
            solver       = arch.get("solver",       "lbfgs"),
            n_jobs       = arch.get("n_jobs",       -1),
            random_state = seed,
        )

    if clf_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed.  "
                "Run:  pip install xgboost"
            ) from exc
        return XGBClassifier(
            n_estimators     = arch.get("n_estimators",     300),
            max_depth        = arch.get("max_depth",        6),
            learning_rate    = arch.get("learning_rate",    0.05),
            subsample        = arch.get("subsample",        0.8),
            colsample_bytree = arch.get("colsample_bytree", 0.8),
            n_jobs           = arch.get("n_jobs",           -1),
            eval_metric      = "mlogloss",
            random_state     = seed,
        )

    if clf_type == "mlp":
        hidden = arch.get("hidden_layer_sizes", [256, 128, 64])
        return MLPClassifier(
            hidden_layer_sizes  = tuple(hidden),
            activation          = arch.get("activation",          "relu"),
            alpha               = arch.get("alpha",               0.0001),
            batch_size          = arch.get("batch_size",          "auto"),
            max_iter            = arch.get("max_iter",            1000),
            learning_rate_init  = arch.get("learning_rate_init",  0.001),
            early_stopping      = arch.get("early_stopping",      True),
            validation_fraction = arch.get("validation_fraction", 0.1),
            n_iter_no_change    = arch.get("n_iter_no_change",    20),
            random_state        = seed,
        )

    raise ValueError(
        f"Unknown classifier type '{clf_type}'. "
        "Valid options: random_forest, svm, logistic_regression, xgboost, mlp"
    )


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(config):
    """
    Build a sklearn Pipeline:  StandardScaler  →  Classifier

    Parameters
    ----------
    config : dict

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", build_classifier(config)),
    ])
