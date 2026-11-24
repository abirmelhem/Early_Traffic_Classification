"""Synthetic pools and mock classifiers only; no research model fitting."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.ml import imbalance_experiments as runner
from src.ml.balancing import balanced_sample_weights, oversample
from src.ml.features import BUDGETS, CLASSES, DESIGN, PREDICTORS
from src.ml.models import xgboost_model


def pool(prefix="fit", counts=(1, 2, 3, 4)):
    return pd.DataFrame([dict(flow_id=f"{prefix}-{c}-{i}", capture_id=f"{prefix}-{c}.pcap",
        candidate_class=c, verification_status="capture_derived_unverified", label_source="synthetic",
        **{col: float(j+i) for j, col in enumerate(PREDICTORS)})
        for c, count in zip(CLASSES, counts) for i in range(count)])


class SyntheticData:
    hashes = {}

    def __init__(self):
        self.cohort = pd.concat([pool(f"g{i}") for i in range(4)], ignore_index=True)

    def partition(self, fold, n, role):
        groups = {"test": [fold-1], "validation": [fold % 4],
                  "train": [i for i in range(4) if i not in (fold-1, fold % 4)]}
        return pd.concat([pool(f"g{i}") for i in groups[role]], ignore_index=True)


class RecordingPipeline:
    def __init__(self, log):
        self.log, self.params = log, {}

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        return self

    def fit(self, X, y, **kwargs):
        self.log.append(("fit", X.copy(), np.asarray(y), kwargs, self.params.copy()))
        return self

    def predict(self, X):
        self.log.append(("predict",))
        return np.repeat(CLASSES[0], len(X))


class ImbalanceTests(unittest.TestCase):
    def baseline_fixture(self, root):
        folder = root / "results/ml/logistic_regression"
        folder.mkdir(parents=True)
        config = dict(model="logistic_regression", grid=[{"C": 1}, {"C": 2}], comparison_identity=dict(
            inputs={}, design=DESIGN, predictors=PREDICTORS, labels=CLASSES, budgets=BUDGETS,
            seed=42, protocol="capture_cv_train_validation_refit_v1"))
        (folder / "configuration.json").write_text(json.dumps(config), encoding="utf-8")
        pd.DataFrame([dict(n_packets=n, fold=f, parameters='{"C":2}') for n in BUDGETS for f in range(1, 5)]).to_csv(folder / "selected_parameters.csv", index=False, lineterminator="\n")
        pd.DataFrame([dict(n_packets=n, fold=f, parameters=json.dumps({"C": c}), validation_macro_f1=c/10)
            for n in BUDGETS for f in range(1, 5) for c in (1, 2)]).to_csv(folder / "validation_search.csv", index=False, lineterminator="\n")
        marker = dict(status="completed", model="logistic_regression", output_sha256={p.name: runner.sha(p) for p in folder.iterdir()})
        (folder / "completed.json").write_text(json.dumps(marker), encoding="utf-8")
        return folder

    def test_frozen_selections_and_explicit_newline_hash_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.baseline_fixture(root)
            path = folder / "selected_parameters.csv"
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            selections, fingerprints, normalized = runner.baseline_parameters(root, "logistic_regression", SyntheticData())
            self.assertEqual(len(selections), 16)
            self.assertTrue(all(p == {"C": 2} for p in selections.values()))
            self.assertEqual(normalized, ["selected_parameters.csv"])
            self.assertEqual(fingerprints[path.relative_to(root).as_posix()], runner.sha(path))

    def test_substantive_baseline_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.baseline_fixture(root)
            path = folder / "selected_parameters.csv"
            path.write_bytes(path.read_bytes().replace(b':2', b':1'))
            with self.assertRaisesRegex(ValueError, "checksum"):
                runner.baseline_parameters(root, "logistic_regression", SyntheticData())

    def test_selection_must_match_validation_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.baseline_fixture(root)
            path = folder / "selected_parameters.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "parameters"] = '{"C":1}'
            frame.to_csv(path, index=False)
            marker = folder / "completed.json"
            done = json.loads(marker.read_text())
            done["output_sha256"][path.name] = runner.sha(path)
            marker.write_text(json.dumps(done))
            with self.assertRaisesRegex(ValueError, "Selection disagrees"):
                runner.baseline_parameters(root, "logistic_regression", SyntheticData())

    def test_retains_every_original_value_and_balances(self):
        original = pool()
        before = original.copy(deep=True)
        result = oversample(original, 43)
        pd.testing.assert_frame_equal(original, before)
        self.assertEqual(result.candidate_class.value_counts().to_dict(), dict.fromkeys(CLASSES, 4))
        self.assertEqual(set(result.flow_id), set(original.flow_id))
        for row in original.itertuples(index=False, name=None):
            matched = result[result.flow_id == row[0]]
            self.assertGreaterEqual(len(matched), 1)
            self.assertTrue(all(r == row for r in matched.itertuples(index=False, name=None)))
        self.assertTrue((result.flow_id.value_counts().reindex(original[original.candidate_class == "FT"].flow_id) == 1).all())

    def test_determinism_order_independence_and_balanced_input(self):
        original = pool()
        pd.testing.assert_frame_equal(oversample(original, 43), oversample(original.sample(frac=1, random_state=2), 43))
        balanced = pool(counts=(2, 2, 2, 2))
        self.assertTrue((oversample(balanced, 43).flow_id.value_counts() == 1).all())

    def test_invalid_pools_fail(self):
        for frame in (pool().iloc[:0], pool()[lambda x: x.candidate_class != "CHAT"], pd.concat([pool(), pool().iloc[:1]]), pool().assign(candidate_class="UNRESOLVED")):
            with self.assertRaises(ValueError):
                oversample(frame, 42)

    def test_weight_formula_uses_fitting_counts(self):
        frame = pool()
        weights = balanced_sample_weights(frame)
        for cls, count in zip(CLASSES, (1, 2, 3, 4)):
            np.testing.assert_allclose(weights[frame.candidate_class == cls], 10/(4*count))
        self.assertAlmostEqual(weights.sum(), len(frame))

    def test_overlap_rejected_before_balancing_or_build(self):
        data = SyntheticData()
        partition = data.partition
        selections = {(n, f): {} for n in BUDGETS for f in range(1, 5)}
        for method in runner.METHODS:
            backend = SimpleNamespace(build=Mock())
            with patch.object(data, "partition", side_effect=lambda f, n, p: partition(f, n, "train" if p == "test" else p)), patch.object(runner, "oversample") as sampling, patch.object(runner, "balanced_sample_weights") as weights, patch.object(runner, "model_module", return_value=backend):
                with self.assertRaisesRegex(ValueError, "leakage"):
                    runner.run_experiment(data, "xgboost", method, selections)
                sampling.assert_not_called()
                weights.assert_not_called()
                backend.build.assert_not_called()

    def test_capture_overlap_with_distinct_flow_ids_rejected(self):
        train, val, test = pool("train"), pool("val"), pool("test")
        test.loc[0, "capture_id"] = train.iloc[0].capture_id
        with self.assertRaisesRegex(ValueError, "leakage"):
            runner.fitting_pool(train, val, test)

    def test_mock_all_models_methods_and_refit_test_barrier(self):
        data = SyntheticData()
        selections = {(n, f): {"frozen": f} for n in BUDGETS for f in range(1, 5)}
        for model in runner.MODELS:
            for method in runner.METHODS:
                log, built = [], []
                def build(params, seed):
                    built.append(params)
                    return RecordingPipeline(log)
                with patch.object(runner, "model_module", return_value=SimpleNamespace(build=build)):
                    tables = runner.run_experiment(data, model, method, selections)
                self.assertEqual([x[0] for x in log], ["fit"]*16 + ["predict"]*16)
                self.assertEqual(built, [selections[(n, f)] for n in BUDGETS for f in range(1, 5)])
                manifest = tables["fitting_manifest"]
                for n in BUDGETS:
                    preds = tables["predictions"].query("n_packets == @n")
                    self.assertEqual(set(preds.flow_id), set(data.cohort.flow_id))
                    self.assertFalse(preds.flow_id.duplicated().any())
                    for f in range(1, 5):
                        fit = manifest[(manifest.n_packets == n) & (manifest.fold == f)]
                        test = data.partition(f, n, "test")
                        self.assertFalse(set(fit.flow_id) & set(test.flow_id))
                        self.assertFalse(set(fit.capture_id) & set(test.capture_id))
                        self.assertTrue((fit.multiplicity >= 1).all())
                        np.testing.assert_array_equal(fit.multiplicity, manifest[(manifest.n_packets == 3) & (manifest.fold == f)].multiplicity)
                for _, X, y, kwargs, params in log[:16]:
                    self.assertEqual(tuple(X.columns), PREDICTORS)
                    if method == "class_weighted" and model == "xgboost":
                        self.assertEqual(set(kwargs), {"classifier__sample_weight"})
                        counts = pd.Series(y).value_counts()
                        np.testing.assert_allclose(kwargs["classifier__sample_weight"], [len(y)/(4*counts[c]) for c in y])
                    elif method == "class_weighted":
                        self.assertEqual(params, {"classifier__class_weight": "balanced"})
                    else:
                        self.assertFalse(kwargs)
                        self.assertEqual(len(set(pd.Series(y).value_counts())), 1)

    def test_xgboost_sample_weights_encoding_and_cpu(self):
        calls = {}
        class FakeXGB:
            def __init__(self, **kwargs): calls["params"] = kwargs
            def fit(self, X, y, sample_weight=None): calls.update(labels=y.copy(), weights=sample_weight.copy())
        frame = pool()
        weights = balanced_sample_weights(frame)
        with patch.object(xgboost_model, "xgb_class", return_value=FakeXGB):
            pipe = xgboost_model.build(xgboost_model.parameter_grid()[0])
            pipe.fit(frame[list(PREDICTORS)], frame.candidate_class, classifier__sample_weight=weights)
        np.testing.assert_array_equal(calls["weights"], weights)
        np.testing.assert_array_equal(calls["labels"], [CLASSES.index(c) for c in frame.candidate_class])
        self.assertEqual(calls["params"]["device"], "cpu")
        self.assertEqual(calls["params"]["tree_method"], "hist")
        self.assertEqual(calls["params"]["n_jobs"], 1)

    def test_invalid_xgboost_weights_fail_before_backend(self):
        for weights in ([1], [1, 1, 1, -1], [1, 1, 1, float("nan")], [[1], [1], [1], [1]]):
            with patch.object(xgboost_model, "xgb_class") as backend:
                with self.assertRaisesRegex(ValueError, "weights"):
                    xgboost_model.EncodedXGBClassifier().fit(np.zeros((4, 7)), CLASSES, sample_weight=weights)
                backend.assert_not_called()

    def test_validate_only_never_constructs_trains_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "load_data", return_value=SyntheticData()), patch.object(runner, "baseline_parameters", return_value=({}, {}, [])), patch.object(runner, "run_experiment", side_effect=AssertionError("training forbidden")), patch.object(runner, "model_module", side_effect=AssertionError("construction forbidden")), contextlib.redirect_stdout(io.StringIO()):
                runner.execute(SimpleNamespace(root=root, model="xgboost", method="class_weighted", validate_only=True))
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_output_refused_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = runner.output_path(root, "xgboost", "random_oversampling")
            output.mkdir(parents=True)
            sentinel = output / "untouched.txt"
            sentinel.write_text("existing work")
            with patch.object(runner, "load_data", side_effect=AssertionError("should not load")):
                with self.assertRaises(FileExistsError):
                    runner.execute(SimpleNamespace(root=root, model="xgboost", method="random_oversampling", validate_only=False))
            self.assertEqual(sentinel.read_text(), "existing work")

    def test_mock_exports_and_historical_outputs_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = [root / f"results/ml/{name}/predictions.csv" for name in ("random_forest", "weighted_exploratory", "oversampling_exploratory")]
            for path in protected:
                path.parent.mkdir(parents=True)
                path.write_bytes(b"historical bytes\r\n")
            hashes = {p.relative_to(root).as_posix(): runner.sha(p) for p in protected}
            selections = {(n, f): {} for n in BUDGETS for f in range(1, 5)}
            with patch.object(runner, "load_data", return_value=SyntheticData()), patch.object(runner, "baseline_parameters", return_value=(selections, hashes, [])), patch.object(runner, "model_module", return_value=SimpleNamespace(build=lambda p, seed: RecordingPipeline([]))), contextlib.redirect_stdout(io.StringIO()):
                runner.execute(SimpleNamespace(root=root, model="random_forest", method="random_oversampling", validate_only=False))
            self.assertTrue(all(runner.sha(root / p) == digest for p, digest in hashes.items()))
            output = runner.output_path(root, "random_forest", "random_oversampling")
            done = json.loads((output / "completed.json").read_text())
            for name, digest in done["output_sha256"].items():
                self.assertEqual(runner.sha(output / name), digest)
            config = json.loads((output / "configuration.json").read_text())
            self.assertEqual(config["protocol"], runner.PROTOCOL)
            self.assertEqual(config["predictors"], list(PREDICTORS))
            for name in ("predictions", "fitting_manifest", "selected_parameters", "fold_class_metrics", "pooled_class_metrics"):
                self.assertTrue((output / f"{name}.csv").is_file())


if __name__ == "__main__":
    unittest.main()
