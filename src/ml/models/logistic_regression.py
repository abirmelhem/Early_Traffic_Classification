from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parameter_grid():
    return [{"C": c} for c in (.001, .01, .1, .3, 1.0, 3.0, 10.0, 100.0)]


def build(params, seed=42):
    if set(params) != {"C"}:
        raise ValueError("Unsupported Logistic Regression parameters")
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scaler", StandardScaler()),
                     ("classifier", LogisticRegression(C=params["C"], solver="lbfgs", max_iter=2000,
                                                       random_state=seed, class_weight=None))])
