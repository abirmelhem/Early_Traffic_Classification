"""Adversarial partition tests and synthetic preprocessing; no research fitting."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import test_imbalance_experiments as fixtures
from src.ml import imbalance_experiments as runner
from src.ml.balancing import balanced_sample_weights
from src.ml.evaluation import select_configuration, test_predictions as score_test
from src.ml.features import BUDGETS, CLASSES, PREDICTORS, predictor_frame


class FakeClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array(CLASSES)
        self.fitted_y_ = np.asarray(y).copy()
        self.sample_weight_ = None if sample_weight is None else sample_weight.copy()
        return self

    def predict(self, X):
        return np.repeat(CLASSES[0], len(X))


class LeakageTests(unittest.TestCase):
    def selections(self):
        return {(n, f): {} for n in BUDGETS for f in range(1, 5)}

    def test_flow_id_index_like_real_loader_is_supported(self):
        data = fixtures.SyntheticData()
        partition = data.partition
        def indexed(fold, n, role):
            return partition(fold, n, role).set_index("flow_id", drop=False)
        with patch.object(data, "partition", side_effect=indexed):
            frames = runner.validated_partitions(data)
        self.assertEqual(len(frames), 48)
        from src.ml.balancing import oversample
        original = fixtures.pool().set_index("flow_id", drop=False)
        self.assertEqual(set(oversample(original, 43).flow_id), set(original.flow_id))

    def test_each_partition_pair_rejects_flow_and_capture_overlap(self):
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            for key in ("flow_id", "capture_id"):
                with self.subTest(left=left, right=right, key=key):
                    data = fixtures.SyntheticData()
                    partition = data.partition
                    def contaminated(fold, n, role):
                        frame = partition(fold, n, role)
                        if role == right:
                            frame.loc[0, key] = partition(fold, n, left).iloc[0][key]
                        return frame
                    with patch.object(data, "partition", side_effect=contaminated), patch.object(runner, "fit_balanced_pipeline") as fitting:
                        with self.assertRaises(ValueError):
                            runner.run_experiment(data, "xgboost", "random_oversampling", self.selections())
                        fitting.assert_not_called()

    def test_cross_n_partition_drift_fails_before_any_fit(self):
        data = fixtures.SyntheticData()
        partition = data.partition
        def swapped(fold, n, role):
            if n == 20 and role in ("train", "validation"):
                role = "validation" if role == "train" else "train"
            return partition(fold, n, role)
        with patch.object(data, "partition", side_effect=swapped), patch.object(runner, "fit_balanced_pipeline") as fitting:
            with self.assertRaisesRegex(ValueError, "across N"):
                runner.run_experiment(data, "xgboost", "class_weighted", self.selections())
            fitting.assert_not_called()

    def test_changed_labels_or_capture_assignments_rejected(self):
        for column, value in (("candidate_class", "MAIL"), ("capture_id", "different.pcap")):
            data = fixtures.SyntheticData()
            partition = data.partition
            def changed(fold, n, role):
                frame = partition(fold, n, role)
                if role == "test":
                    frame.loc[0, column] = value
                return frame
            with patch.object(data, "partition", side_effect=changed), patch.object(runner, "fit_balanced_pipeline") as fitting:
                with self.assertRaises(ValueError):
                    runner.run_experiment(data, "xgboost", "class_weighted", self.selections())
                fitting.assert_not_called()

    def test_resampler_cannot_inject_omit_or_modify_rows(self):
        original = fixtures.pool()
        def inject(frame, seed):
            result = runner.oversample_original(frame, seed)
            result.loc[0, "flow_id"] = "outer-test-observation"
            return result
        def omit(frame, seed):
            return runner.oversample_original(frame, seed)[lambda x: x.flow_id != frame.iloc[0].flow_id]
        def mutate(frame, seed):
            result = runner.oversample_original(frame, seed)
            result.loc[0, "mean_length"] += 999
            return result
        from src.ml.balancing import oversample
        for sampler in (inject, omit, mutate):
            with patch.object(runner, "oversample_original", oversample, create=True), patch.object(runner, "oversample", side_effect=sampler), patch.object(runner, "model_module") as backend:
                with self.assertRaisesRegex(ValueError, "Resampling"):
                    runner.fit_balanced_pipeline(original, "xgboost", "random_oversampling", {}, 42)
                backend.assert_not_called()

    def test_historical_selection_fits_training_preprocessing_only(self):
        train, val = fixtures.pool("train"), fixtures.pool("val", (4, 3, 2, 1))
        train["mean_length"], val["mean_length"] = 2.0, 100000.0
        before = val.copy(deep=True)
        pipes = []
        def builder(params, seed):
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", FakeClassifier())])
            pipes.append(pipe)
            return pipe
        with patch.object(runner, "oversample", side_effect=AssertionError("No balancing in historical selection")), patch.object(runner, "balanced_sample_weights", side_effect=AssertionError("No weighting in historical selection")):
            select_configuration(train, val, builder, [{"tag": 1}, {"tag": 2}], 42)
        for pipe in pipes:
            self.assertEqual(pipe.named_steps["scaler"].mean_[0], 2.0)
            self.assertEqual(pipe.named_steps["imputer"].statistics_[0], 2.0)
            self.assertEqual(len(pipe.named_steps["classifier"].fitted_y_), len(train))
        pd.testing.assert_frame_equal(val, before)

    def test_training_only_balancing_then_fresh_combined_refit(self):
        train, val = fixtures.pool("train"), fixtures.pool("val", (4, 1, 1, 1))
        train["mean_length"], val["mean_length"] = 2.0, 100000.0
        before = val.copy(deep=True)
        test = fixtures.pool("test")
        for method in runner.METHODS:
            pipes = []
            def build(params, seed):
                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", FakeClassifier())])
                pipes.append(pipe)
                return pipe
            with patch.object(runner, "model_module", return_value=SimpleNamespace(build=build)):
                selection_fit, sampled, weights = runner.fit_balanced_pipeline(train, "xgboost", method, {}, 43)
                self.assertEqual(set(sampled.flow_id), set(train.flow_id))
                self.assertEqual(selection_fit.named_steps["scaler"].mean_[0], 2.0)
                predictions = selection_fit.predict(predictor_frame(val))
                self.assertEqual(len(predictions), len(val))
                pd.testing.assert_frame_equal(val, before)
                combined = runner.fitting_pool(train, val, test)
                final, refit_rows, final_weights = runner.fit_balanced_pipeline(combined, "xgboost", method, {}, 43)
            self.assertIsNot(selection_fit, final)
            self.assertGreater(final.named_steps["scaler"].mean_[0], 2.0)
            self.assertEqual(set(refit_rows.flow_id), set(train.flow_id) | set(val.flow_id))
            self.assertFalse(set(refit_rows.flow_id) & set(test.flow_id))
            if method == "class_weighted":
                np.testing.assert_allclose(weights, balanced_sample_weights(train))
                np.testing.assert_allclose(final_weights, balanced_sample_weights(combined))
                self.assertNotEqual(weights[0], final_weights[0])
            else:
                self.assertEqual(len(sampled), 4*4)
                self.assertEqual(len(refit_rows), 4*5)

    def test_test_scoring_rejects_flow_or_capture_overlap_before_predict(self):
        test = fixtures.pool("test")
        for ids, captures in (([test.iloc[0].flow_id], []), ([], [test.iloc[0].capture_id])):
            backend = fixtures.RecordingPipeline([])
            with self.assertRaisesRegex(ValueError, "overlaps"):
                score_test(backend, test, ids, captures)
            self.assertEqual(backend.log, [])

    def test_mock_predictions_preserve_exact_original_test_rows(self):
        data = fixtures.SyntheticData()
        before = data.cohort.copy(deep=True)
        for method in runner.METHODS:
            with patch.object(runner, "model_module", return_value=SimpleNamespace(build=lambda p, seed: fixtures.RecordingPipeline([]))):
                tables = runner.run_experiment(data, "xgboost", method, self.selections())
            for n in BUDGETS:
                for fold in (1, 2, 3, 4):
                    original = data.partition(fold, n, "test")
                    pred = tables["predictions"].query("n_packets == @n and fold == @fold")
                    expected = original[["flow_id", "capture_id", "candidate_class", "verification_status", "label_source"]].rename(columns={"candidate_class": "y_true"}).reset_index(drop=True)
                    pd.testing.assert_frame_equal(pred[list(expected.columns)].reset_index(drop=True), expected)
                    self.assertFalse(pred.flow_id.duplicated().any())
                    self.assertEqual(pred.y_true.value_counts().to_dict(), original.candidate_class.value_counts().to_dict())
            pd.testing.assert_frame_equal(data.cohort, before)


if __name__ == "__main__":
    unittest.main()
