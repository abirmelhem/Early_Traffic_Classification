"""Readiness audit tests; no splitting or model fitting."""
import unittest

import pandas as pd

from src.analysis.ml_readiness import (
    BUDGETS, PRIMARY, capture_audit, common_cohort, duplicate_audit,
    feature_audit, split_feasibility,
)


class ReadinessTest(unittest.TestCase):
    def setUp(self):
        self.flows = pd.DataFrame([
            dict(flow_id=str(i), capture_id=f"folder/chat{i}.pcap", candidate_class="CHAT",
                 application="AIM", recording_group="", verification_status="capture_derived_unverified",
                 packet_count=20, protocol="TCP") for i in range(4)])
        self.features = pd.DataFrame([
            dict(flow_id=row.flow_id, capture_id=row.capture_id, candidate_class=row.candidate_class,
                 n_packets=n, protocol="TCP", **{k: 1.0 for k in PRIMARY})
            for row in self.flows.itertuples() for n in BUDGETS])

    def test_common_cohort_excludes_unresolved_and_incomplete(self):
        self.flows.loc[0, "candidate_class"] = "UNRESOLVED"
        features = self.features[~((self.features.flow_id == "1") & (self.features.n_packets == 20))]
        cohort = common_cohort(self.flows, features)
        self.assertEqual(set(cohort.flow_id), {"2", "3"})
        self.assertEqual(len(cohort), 2)

    def test_feature_allowlist_excludes_future_and_metadata(self):
        flows = self.flows.assign(forward_count=1, duration_seconds=2, eligible_n20=1)
        audit = feature_audit(flows, self.features)
        self.assertEqual(set(audit.query("recommendation == 'PRIMARY'").column), set(PRIMARY))
        self.assertEqual(audit.set_index("column").loc["packet_count", "recommendation"], "EXCLUDE")
        self.assertEqual(audit.set_index("column").loc["protocol", "recommendation"], "SECONDARY_ONLY")

    def test_unknown_column_requires_review(self):
        with self.assertRaisesRegex(ValueError, "Unaudited column"):
            feature_audit(self.flows.assign(suspicious_numeric=42), self.features)

    def test_recording_groups_not_invented(self):
        manifest = self.flows[["capture_id", "candidate_class", "application", "recording_group", "verification_status"]]
        table = capture_audit(self.flows, self.flows, manifest)
        self.assertTrue(table.recording_group.eq("").all())
        self.assertTrue(table.relationship_evidence.str.contains("NOT a session").all())
        self.assertTrue(table.possible_related_captures.ne("").all())

    def test_manifest_mismatch_rejected(self):
        manifest = self.flows[["capture_id", "candidate_class", "application", "recording_group", "verification_status"]].copy()
        manifest.loc[0, "candidate_class"] = "FT"
        with self.assertRaisesRegex(ValueError, "Manifest disagrees"):
            capture_audit(self.flows, self.flows, manifest)

    def test_feasibility_never_produces_partitions(self):
        cohort = pd.concat([self.flows] * 25, ignore_index=True)
        table = split_feasibility(cohort).set_index("candidate_class")
        self.assertEqual(table.loc["CHAT", "screened_options"], 12)
        self.assertEqual(table.loc["CHAT", "documented_recording_groups"], 0)
        self.assertEqual(table.loc["CHAT", "verified_session_split"], "BLOCKED_missing_independence_evidence")
        self.assertNotIn("partition", table.columns)
        self.assertEqual(table.loc["MAIL", "screened_options"], 0)

    def test_similarity_is_same_n_and_not_deduplicated(self):
        original = len(self.features)
        table = duplicate_audit(self.features)
        self.assertEqual(len(self.features), original)
        self.assertEqual(len(table), 8)  # four N, exact and coarsened
        self.assertTrue(table.matching_flows.eq(4).all())
        self.assertTrue(table.captures.eq(4).all())

    def test_single_capture_screen_can_fail_while_cv_capacity_exists(self):
        small = pd.concat([self.flows.assign(capture_id=lambda x: x.capture_id + str(i))
                           for i in range(3)], ignore_index=True)
        small = pd.concat([small] * 5, ignore_index=True)
        row = split_feasibility(small).set_index("candidate_class").loc["CHAT"]
        self.assertEqual(row.screened_options, 0)
        self.assertEqual(row.four_fold_capacity_min_test_flows, 15)
        self.assertEqual(row.common_flows, 60)


if __name__ == "__main__":
    unittest.main()
