import numpy as np
from loguru import logger

try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

from config import LGBM_PARAMS


class LGBMModel:

    def __init__(self, task="classification", params=None):
        if not LGBM_OK:
            raise ImportError("pip install lightgbm")

        self.task = task

        # Merge params FIRST
        self.params = {
            **LGBM_PARAMS,
            **(params or {}),
        }

        # FORCE GPU (diagnostic requirement)
        self.params["device"] = "gpu"

        # Stability defaults
        self.params.setdefault("gpu_platform_id", 0)
        self.params.setdefault("gpu_device_id", 0)
        self.params.setdefault("max_bin", 255)

        logger.info(f"LGBM device = {self.params.get('device')}")

        # TASK
        if self.task == "classification":
            self.params["objective"] = "multiclass"
            self.params["num_class"] = 3
            self.params["metric"] = "multi_logloss"

        elif self.task == "regression":
            self.params["objective"] = "regression"
            self.params["metric"] = "rmse"

        else:
            raise ValueError(f"Unknown task: {self.task}")

        self.model = None
        self.feature_names_ = None

    def fit(self, X, y):
        if self.task == "classification":
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)

        self.model.fit(X, y)
        self.feature_names_ = getattr(self.model, "feature_name_", None)

    def predict(self, X):
        return self.model.predict(X)

    def predict_class(self, X):
        if self.task != "classification":
            raise ValueError("predict_class only for classification")
        return np.argmax(self.model.predict_proba(X), axis=1)

    def feature_importance(self):
        return self.model.feature_importances_

    def save(self, path):
        self.model.booster_.save_model(path)

    def load(self, path):
        self.model = lgb.Booster(model_file=path)