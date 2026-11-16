from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def parameter_grid():
    return [dict(n_estimators=n, max_depth=d, min_samples_leaf=l)
            for n, d, l in product((100, 200), (None, 10), (1, 3))]


def build(params, seed=42):
    if set(params) != {"n_estimators", "max_depth", "min_samples_leaf"}:
        raise ValueError("Unsupported Random Forest parameters")
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("classifier", RandomForestClassifier(**params, random_state=seed, n_jobs=1,
                                                            class_weight=None))])
