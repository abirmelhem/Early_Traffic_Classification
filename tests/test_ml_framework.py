"""Synthetic fixtures and mocked estimators only; never fit a research classifier."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml import compare_models, run_experiments
from src.ml.data_loader import load_data
from src.ml.evaluation import refit_selected, select_configuration
from src.ml.features import BUDGETS, CLASSES, DESIGN, PREDICTORS, predictor_frame
from src.ml.metrics import aggregate_predictions, calculate_metrics
from src.ml.models import model_module
from src.ml.models import xgboost_model


class DummyEstimator(ClassifierMixin, BaseEstimator):
    """Non-learning estimator for synthetic preprocessing-isolation tests."""
    def fit(self, X, y):
        self.classes_ = np.array(CLASSES)
        return self

    def predict(self, X):
        return np.repeat(CLASSES[0], len(X))


class MLFrameworkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = []
        for cls in CLASSES:
            for i in range(4):
                self.source.append(dict(flow_id=f"{cls}-{i}", capture_id=f"{cls}/{i}.pcap",
                                        candidate_class=cls, verification_status="capture_derived_unverified",
                                        label_source="synthetic fixture", recording_group="", packet_count=20))
        self.cohort = pd.DataFrame(self.source)
        self.feature_rows = [dict(row, n_packets=n, **{col: float(i+1) for col in PREDICTORS})
                             for i, row in enumerate(self.source) for n in BUDGETS]
        assignments = []
        for fold in range(1, 5):
            for row in self.source:
                index = int(row["flow_id"].split("-")[-1])
                role = "test" if index == fold-1 else "validation" if index == fold % 4 else "train"
                assignments.append(dict(row, design=DESIGN, round=fold, partition=role))
        self.assignments = pd.DataFrame(assignments)
        self.write("data/processed/capture_labeled/all_flows_capture_labeled.csv", self.cohort)
        self.write("data/processed/capture_labeled/all_features_capture_labeled.csv", pd.DataFrame(self.feature_rows))
        self.write("results/ml_readiness/common_cohort_manifest.csv", self.cohort)
        self.write("results/ml_readiness/feature_audit.csv", pd.DataFrame({"column": PREDICTORS, "recommendation": "PRIMARY"}))
        self.write("results/capture_splits/flow_partition_manifest.csv", self.assignments)
        observations = pd.DataFrame([dict(row, n_packets=n) for row in assignments for n in BUDGETS])
        self.write("results/capture_splits/observation_partition_manifest.csv", observations)
        self.write("results/capture_splits/capture_partition_manifest.csv", self.assignments.drop(columns="flow_id"))

    def write(self, relative, frame):
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, index=False)

    def test_loading_and_same_cohort_all_n(self):
        data = load_data(self.root)
        self.assertEqual(len(data.cohort), 16)
        for fold in (1, 2, 3, 4):
            sets = [set(data.partition(fold, n, "test").flow_id) for n in BUDGETS]
            self.assertTrue(all(s == sets[0] for s in sets))
            self.assertEqual(len(sets[0]), 4)

    def test_capture_overlap_rejected(self):
        self.assignments.loc[0, "capture_id"] = "MAIL/0.pcap"
        self.write("results/capture_splits/flow_partition_manifest.csv", self.assignments)
        with self.assertRaises(ValueError):
            load_data(self.root)

    def test_duplicate_flow_rejected(self):
        self.write("results/capture_splits/flow_partition_manifest.csv", pd.concat([self.assignments, self.assignments.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_data(self.root)

    def test_n_partition_mismatch_rejected(self):
        path = self.root / "results/capture_splits/observation_partition_manifest.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        frame.loc[0, "partition"] = "train" if frame.loc[0, "partition"] != "train" else "test"
        self.write(path.relative_to(self.root), frame)
        with self.assertRaisesRegex(ValueError, "N representations"):
            load_data(self.root)

    def test_missing_representation_and_unresolved_rejected(self):
        self.write("data/processed/capture_labeled/all_features_capture_labeled.csv", pd.DataFrame(self.feature_rows[1:]))
        with self.assertRaisesRegex(ValueError, "Missing N"):
            load_data(self.root)
        self.cohort.loc[0, "candidate_class"] = "UNRESOLVED"
        self.write("results/ml_readiness/common_cohort_manifest.csv", self.cohort)
        with self.assertRaisesRegex(ValueError, "Common cohort"):
            load_data(self.root)

    def test_whitelist_rejects_numeric_metadata(self):
        frame = pd.DataFrame(self.feature_rows).assign(packet_count=1000000, target_code=42)
        self.assertEqual(tuple(predictor_frame(frame).columns), PREDICTORS)
        with self.assertRaisesRegex(ValueError, "whitelist"):
            predictor_frame(frame, columns=(*PREDICTORS, "packet_count"))
        self.write("results/ml_readiness/feature_audit.csv", pd.DataFrame({"column": [*PREDICTORS, "packet_count"], "recommendation": "PRIMARY"}))
        with self.assertRaisesRegex(ValueError, "Feature audit"):
            load_data(self.root)

    def test_model_construction_and_grids_without_fit(self):
        for name in ("logistic_regression", "random_forest"):
            module = model_module(name)
            self.assertEqual(len(module.parameter_grid()), 8)
            pipe = module.build(module.parameter_grid()[0], 42)
            self.assertIsInstance(pipe.named_steps["imputer"], SimpleImputer)
            self.assertEqual(pipe.named_steps["classifier"].random_state, 42)
            self.assertIsNone(pipe.named_steps["classifier"].class_weight)
            self.assertEqual("scaler" in pipe.named_steps, name == "logistic_regression")
        self.assertEqual(len(xgboost_model.parameter_grid()), 8)
        with patch.object(xgboost_model, "xgb_class", return_value=DummyEstimator):
            pipe = xgboost_model.build(xgboost_model.parameter_grid()[0], 42)
            self.assertIsInstance(pipe.named_steps["classifier"], xgboost_model.EncodedXGBClassifier)

    def test_xgboost_cpu_encoding_with_mock_backend(self):
        calls = {}
        class FakeXGB:
            def __init__(self, **kwargs): calls["params"] = kwargs
            def fit(self, X, y): calls["encoded"] = y.tolist()
            def predict(self, X): return np.arange(len(X))
        with patch.object(xgboost_model, "xgb_class", return_value=FakeXGB):
            estimator = xgboost_model.EncodedXGBClassifier()
            estimator.fit(np.zeros((4, 7)), np.array(CLASSES))
            self.assertEqual(calls["encoded"], [0, 1, 2, 3])
            self.assertEqual(estimator.predict(np.zeros((4, 7))).tolist(), list(CLASSES))
            self.assertEqual(calls["params"]["tree_method"], "hist")
            self.assertEqual(calls["params"]["device"], "cpu")
            self.assertEqual(calls["params"]["objective"], "multi:softprob")

    def test_actual_preprocessing_isolation_and_fresh_refit(self):
        data = load_data(self.root)
        train = data.partition(1, 3, "train")
        validation = data.partition(1, 3, "validation")
        train.loc[:, "mean_length"] = 2.0
        validation.loc[:, "mean_length"] = 1000.0
        constructed = []
        def builder(params, seed):
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler()), ("classifier", DummyEstimator())])
            constructed.append(pipe)
            return pipe
        params, trials = select_configuration(train, validation, builder, [{"tag": "b"}, {"tag": "a"}], 42)
        self.assertEqual(params, {"tag": "a"})
        self.assertEqual(trials[0]["validation_macro_f1"], trials[1]["validation_macro_f1"])
        for pipe in constructed:
            self.assertEqual(pipe.named_steps["scaler"].mean_[0], 2.0)
            self.assertEqual(pipe.named_steps["imputer"].statistics_[0], 2.0)
        final = refit_selected(train, validation, builder, params, 42)
        self.assertIsNot(final, constructed[0])
        self.assertAlmostEqual(final.named_steps["scaler"].mean_[0], (8*2+4*1000)/12)

    def test_metrics_schema_and_fixed_class_order(self):
        summary, classes, matrix = calculate_metrics(CLASSES, CLASSES)
        self.assertEqual(summary["macro_f1"], 1.0)
        self.assertEqual(classes.candidate_class.tolist(), list(CLASSES))
        self.assertEqual(matrix.tolist(), np.eye(4, dtype=int).tolist())
        predictions = pd.DataFrame(dict(flow_id=list("abcd"), n_packets=3, y_true=CLASSES, y_pred=CLASSES))
        pooled, _, confusion = aggregate_predictions(predictions)
        self.assertEqual(pooled.support.iloc[0], 4)
        self.assertEqual(len(confusion), 16)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_predictions(pd.concat([predictions, predictions]))

    def test_mock_execution_selection_refit_and_test_barrier(self):
        data = load_data(self.root)
        calls = []
        class SpyPipeline:
            def fit(self, X, y): calls.append(("fit", tuple(X.index))); return self
            def predict(self, X): calls.append(("predict", tuple(X.index))); return np.repeat("CHAT", len(X))
        fake = SimpleNamespace(parameter_grid=lambda: [{"tag": "b"}, {"tag": "a"}], build=lambda params, seed: SpyPipeline())
        with patch.object(run_experiments, "model_module", return_value=fake) as dispatch:
            tables = run_experiments.run_model(data, "random_forest", [3])
        dispatch.assert_called_once_with("random_forest")
        self.assertEqual(len(tables["predictions"]), 16)
        self.assertEqual(len(tables["pooled_metrics"]), 1)
        # Eight candidate training-only fits, then four fresh train+validation fits.
        fits = [ids for event, ids in calls if event == "fit"]
        self.assertEqual(len(fits), 12)
        for fold in range(1, 5):
            tr = set(data.partition(fold, 3, "train").flow_id)
            va = set(data.partition(fold, 3, "validation").flow_id)
            te = set(data.partition(fold, 3, "test").flow_id)
            self.assertEqual(set(fits[2*(fold-1)]), tr)
            self.assertEqual(set(fits[8+fold-1]), tr|va)
            self.assertFalse(te & set(fits[8+fold-1]))
        last_fit = max(i for i, (event, _) in enumerate(calls) if event == "fit")
        self.assertEqual([event for event, _ in calls[last_fit+1:]], ["predict"]*4)

    def test_validate_only_cannot_call_training(self):
        data = load_data(self.root)
        args = run_experiments.parse_args(["--model", "xgboost", "--validate-only", "--root", str(self.root)])
        with patch.object(run_experiments, "load_data", return_value=data), patch.object(run_experiments, "run_model", side_effect=AssertionError("Training forbidden")), patch.object(run_experiments, "model_module", side_effect=AssertionError("Backend forbidden")), contextlib.redirect_stdout(io.StringIO()):
            run_experiments.execute(args)
        self.assertFalse((self.root / "results/ml").exists())

    def test_comparison_with_no_results_produces_no_files(self):
        with self.assertWarns(UserWarning), contextlib.redirect_stdout(io.StringIO()):
            result = compare_models.compare(self.root)
        self.assertIsNone(result)
        self.assertFalse((self.root / "results/ml").exists())

    def fake_results(self, name, identity=None):
        folder = self.root / "results/ml" / name
        folder.mkdir(parents=True)
        prediction = pd.DataFrame(dict(flow_id=list("abcd"), capture_id=list("abcd"), fold=1, n_packets=3, y_true=CLASSES, y_pred=CLASSES))
        summary, classes, _ = aggregate_predictions(prediction)
        prediction.to_csv(folder / "predictions.csv", index=False)
        summary.to_csv(folder / "pooled_metrics.csv", index=False)
        classes.to_csv(folder / "pooled_class_metrics.csv", index=False)
        (folder / "configuration.json").write_text(json.dumps(dict(comparison_identity=identity or {"synthetic": True})))
        completion = dict(status="completed", model=name, output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()})
        (folder / "completed.json").write_text(json.dumps(completion))

    def test_partial_comparison_and_mismatch_rejection(self):
        self.fake_results("random_forest")
        results, missing = compare_models.load_results(self.root)
        self.assertEqual(list(results), ["random_forest"])
        self.assertEqual(set(missing), {"logistic_regression", "xgboost"})
        self.fake_results("logistic_regression", {"different": True})
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            compare_models.load_results(self.root)

    def test_existing_output_is_never_overwritten(self):
        data = load_data(self.root)
        (self.root / "results/ml/random_forest").mkdir(parents=True)
        args = run_experiments.parse_args(["--model", "random_forest", "--root", str(self.root)])
        with patch.object(run_experiments, "load_data", return_value=data), patch.object(run_experiments, "run_model", side_effect=AssertionError("Training forbidden")):
            with self.assertRaises(FileExistsError): run_experiments.execute(args)

    def test_mock_end_to_end_output_schemas_and_partial_comparison(self):
        data = load_data(self.root)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*.csv")}
        class MockPipeline:
            def fit(self, X, y): return self
            def predict(self, X): return np.repeat("CHAT", len(X))
        fake = SimpleNamespace(parameter_grid=lambda: [{"tag": "a"}], build=lambda params, seed: MockPipeline())
        args = run_experiments.parse_args(["--model", "random_forest", "--n", "3", "--root", str(self.root)])
        with patch.object(run_experiments, "load_data", return_value=data), patch.object(run_experiments, "model_module", return_value=fake), contextlib.redirect_stdout(io.StringIO()):
            run_experiments.execute(args)
        folder = self.root / "results/ml/random_forest"
        self.assertTrue((folder / "completed.json").is_file())
        self.assertEqual(len(pd.read_csv(folder / "predictions.csv")), 16)
        config = json.loads((folder / "configuration.json").read_text())
        self.assertEqual(config["comparison_identity"]["protocol"], run_experiments.PROTOCOL_VERSION)
        self.assertEqual(config["comparison_identity"]["labels"], list(CLASSES))
        self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items()))
        with self.assertWarns(UserWarning), contextlib.redirect_stdout(io.StringIO()):
            compared = compare_models.compare(self.root)
        self.assertEqual(len(compared), 1)
        self.assertTrue((self.root / "results/ml/comparisons/macro_f1_by_n.png").is_file())
        # Checksums detect tampering with completed results rather than plotting it.
        with (folder / "pooled_metrics.csv").open("a") as handle: handle.write("corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            compare_models.load_results(self.root)


if __name__ == "__main__":
    unittest.main()
