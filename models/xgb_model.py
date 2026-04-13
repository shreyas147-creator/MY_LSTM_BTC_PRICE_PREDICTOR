import numpy as np
from loguru import logger

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

from config import XGB_PARAMS


class XGBModel:

    def __init__(self, task: str = "classification", params: dict = None):
        if not XGB_OK:
            raise ImportError("pip install xgboost")

        self.task = task

        # Merge params FIRST
        self.params = {
            **XGB_PARAMS,
            **(params or {}),
        }

        # FORCE CUDA (diagnostic requirement)
        self.params["device"] = "cuda"
        self.params["tree_method"] = "hist"

        logger.info(f"XGB device = {self.params.get('device')}")

        # TASK
        if task == "classification":
            self.params["objective"] = "multi:softprob"
            self.params["num_class"] = 3
            self.params["eval_metric"] = "mlogloss"
        else:
            self.params["objective"] = "reg:squarederror"
            self.params["eval_metric"] = "rmse"

        self.model = None

    def fit(self, X, y):
        self.model = xgb.XGBClassifier(**self.params) if self.task == "classification" \
            else xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_class(self, X):
        if self.task != "classification":
            raise ValueError("predict_class only for classification")
        return np.argmax(self.model.predict_proba(X), axis=1)

    def feature_importance(self):
        return self.model.feature_importances_

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        if self.task == "classification":
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        self.model.load_model(path)