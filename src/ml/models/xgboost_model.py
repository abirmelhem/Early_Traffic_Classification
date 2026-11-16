"""CPU-only multiclass backend; optional dependency loaded only on construction."""
from itertools import product
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..features import CLASSES


def parameter_grid():
    return [dict(n_estimators=n, max_depth=d, learning_rate=r)
            for n, d, r in product((100, 200), (3, 5), (.05, .1))]


def xgb_class():
    try:
        from xgboost import XGBClassifier
        from xgboost import __version__
    except ImportError as exc:
        raise ImportError('XGBoost is optional. Install manually: python -m pip install "xgboost>=2.0"') from exc
    if int(__version__.split(".")[0]) < 2:
        raise ImportError('CPU device option requires XGBoost >=2.0. Install manually: python -m pip install "xgboost>=2.0"')
    return XGBClassifier


class EncodedXGBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        labels = np.asarray(y)
        if set(labels) != set(CLASSES):
            raise ValueError("XGBoost training requires all four original class labels")
        mapping = {name: i for i, name in enumerate(CLASSES)}
        self.estimator_ = xgb_class()(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                     learning_rate=self.learning_rate, random_state=self.random_state,
                                     objective="multi:softprob", num_class=len(CLASSES),
                                     tree_method="hist", device="cpu", n_jobs=1,
                                     eval_metric="mlogloss", subsample=1.0, colsample_bytree=1.0)
        self.estimator_.fit(X, np.array([mapping[name] for name in labels], dtype=int))
        self.classes_ = np.array(CLASSES)
        return self

    def predict(self, X):
        return self.classes_[np.asarray(self.estimator_.predict(X), dtype=int)]


def build(params, seed=42):
    if set(params) != {"n_estimators", "max_depth", "learning_rate"}:
        raise ValueError("Unsupported XGBoost parameters")
    xgb_class()  # Fail before a real run starts if the optional dependency is absent.
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("classifier", EncodedXGBClassifier(**params, random_state=seed))])
