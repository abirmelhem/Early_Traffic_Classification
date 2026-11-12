"""Adversarial partition tests, using synthetic captures and no estimators."""
import unittest

import pandas as pd

from src.analysis.capture_splits import (
    BUDGETS, DESIGNS, PRIMARY, TARGETS, boundary_warnings,
    build_capture_assignments, subset_near, validate_assignments,
)


class CaptureSplitTest(unittest.TestCase):
    def setUp(self):
        records = []
        for cls in TARGETS:
            for i, count in enumerate([20, 25, 30, 35]):
                for j in range(count):
                    records.append(dict(flow_id=f"{cls}-{i}-{j}", capture_id=f"{cls}/{i}.pcap",
                                        candidate_class=cls))
        self.cohort = pd.DataFrame(records)
        self.features = pd.DataFrame([dict(row, n_packets=n, protocol="TCP",
                                           **{k: float(i+1) for k in PRIMARY})
                                      for i, row in enumerate(records) for n in BUDGETS])
        self.captures = build_capture_assignments(self.cohort)
        self.flows = self.captures.drop(columns="flows").merge(self.cohort, on=["capture_id", "candidate_class"])
        self.obs = self.flows.merge(self.features[["flow_id", "n_packets"]], on="flow_id")

    def validate(self):
        validate_assignments(self.cohort, self.features, self.captures, self.flows, self.obs)

    def test_all_designs_have_explicit_three_way_separation(self):
        self.validate()
        self.assertEqual(len(self.flows), sum(DESIGNS.values()) * len(self.cohort))
        for _, group in self.flows.groupby(["design", "round", "partition"]):
            self.assertEqual(set(group.candidate_class), set(TARGETS))

    def test_reproducible_under_input_reordering(self):
        shuffled = self.cohort.sample(frac=1, random_state=7)
        pd.testing.assert_frame_equal(self.captures, build_capture_assignments(shuffled))

    def test_outer_tests_cover_each_flow_once(self):
        for design, folds in DESIGNS.items():
            test = self.flows[(self.flows.design == design) & (self.flows.partition == "test")]
            if folds > 1:
                self.assertEqual(set(test.flow_id), set(self.cohort.flow_id))
                self.assertFalse(test.flow_id.duplicated().any())

    def test_capture_overlap_rejected(self):
        self.captures = pd.concat([self.captures, self.captures.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Capture overlap"):
            self.validate()

    def test_flow_crossing_capture_rejected(self):
        self.flows.loc[0, "partition"] = "test" if self.flows.loc[0, "partition"] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "crosses capture"):
            self.validate()

    def test_n_crossing_partition_rejected(self):
        self.obs.loc[0, "partition"] = "test" if self.obs.loc[0, "partition"] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "N representations cross"):
            self.validate()

    def test_missing_observation_rejected(self):
        self.obs = self.obs.iloc[1:]
        with self.assertRaisesRegex(ValueError, "N representations"):
            self.validate()

    def test_duplicate_flow_rejected(self):
        self.flows = pd.concat([self.flows, self.flows.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Flow overlap"):
            self.validate()

    def test_observation_capture_metadata_rejected(self):
        self.obs.loc[0, "capture_id"] = "wrong_capture"
        with self.assertRaisesRegex(ValueError, "Observation metadata"):
            self.validate()

    def test_tampered_capture_count_rejected(self):
        self.captures.loc[0, "flows"] += 1
        with self.assertRaisesRegex(ValueError, "Capture counts"):
            self.validate()

    def test_mail_ft_retain_two_training_captures_in_four_fold(self):
        for cls in ["MAIL", "FT"]:
            rows = self.captures[(self.captures.design == "outer4_inner_holdout") & (self.captures.candidate_class == cls)]
            for _, r in rows.groupby("round"):
                self.assertEqual(r.partition.value_counts().to_dict(), {"train": 2, "validation": 1, "test": 1})

    def test_infeasible_support_fails_without_flow_fallback(self):
        with self.assertRaisesRegex(ValueError, "No capture-disjoint allocation"):
            subset_near([("a", 3), ("b", 4)], 5, minimum=10, reserve_flows=20)

    def test_duplicate_warnings_cross_boundaries_without_removal(self):
        features = self.features.iloc[:8].copy()  # two flows; same feature vectors
        for col in PRIMARY:
            features[col] = 1.0
        assignments = pd.DataFrame([
            dict(design="example", round=1, flow_id=features.flow_id.iloc[0], capture_id="a", candidate_class="CHAT", partition="train"),
            dict(design="example", round=1, flow_id=features.flow_id.iloc[-1], capture_id="b", candidate_class="CHAT", partition="test")])
        warnings = boundary_warnings(features, assignments)
        self.assertEqual(len(warnings), 8)
        self.assertTrue(warnings.partitions.eq("test;train").all())
        self.assertEqual(len(features), 8)


if __name__ == "__main__":
    unittest.main()
