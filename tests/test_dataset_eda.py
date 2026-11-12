"""Synthetic checks of EDA denominators, identity alignment, and immutable inputs."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from src.analysis import dataset_eda as eda


class DatasetEDATest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "data/processed/capture_labeled"
        self.folder.mkdir(parents=True)
        self.flows, self.features = [], []
        for fid, count, label, proto in [("01", 20, "CHAT", "TCP"), ("02", 2, "CHAT", "UDP"), ("03", 5, "", "UDP")]:
            base = dict(flow_id=fid, capture_id=f"NonVPN-PCAPs-01/{fid}.pcap", protocol=proto,
                        candidate_class=label, label_source="fixture", recording_group="",
                        verification_status="capture_derived_unverified" if label else "unresolved")
            self.flows.append(dict(base, packet_count=count, **{f"eligible_n{n}": int(count >= n) for n in eda.BUDGETS}))
            for n in eda.BUDGETS:
                if count >= n:
                    self.features.append(dict(base, n_packets=n, time_to_n=n / 1000, mean_length=100))
        self.write()

    def write(self):
        for kind, rows in [("flows", self.flows), ("features", self.features)]:
            with (self.folder / f"all_{kind}_capture_labeled.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)

    def tables(self):
        _, flows, features = eda.load_inputs(self.root)
        return eda.summarize(flows, features)

    def test_unique_flow_class_totals_not_feature_rows(self):
        table = self.tables()["class_distribution"].set_index("candidate_class")
        self.assertEqual(table.unique_flows.sum(), 3)
        self.assertEqual(table.loc["CHAT", "unique_flows"], 2)
        self.assertEqual(table.loc["UNRESOLVED", "unique_flows"], 1)
        self.assertEqual(table.loc["BROWSING", "unique_flows"], 0)
        self.assertAlmostEqual(table.dataset_pct.sum(), 100)

    def test_eligibility_matches_features_and_correct_denominators(self):
        _, flows, features = eda.load_inputs(self.root)
        table = eda.summarize(flows, features)["class_by_n"]
        for n in eda.BUDGETS:
            self.assertEqual(table.loc[table.n_packets == n, "eligible_flows"].sum(), (features.n_packets == n).sum())
        chat = table.query("candidate_class == 'CHAT' and n_packets == 20").iloc[0]
        self.assertEqual(chat.eligible_flows, 1)
        self.assertEqual(chat.coverage_pct, 50)
        self.assertEqual(chat.eligible_cohort_pct, 100)
        absent = table.query("candidate_class == 'BROWSING'").coverage_pct
        self.assertTrue(absent.isna().all())

    def test_duplicate_flows_rejected(self):
        self.flows.append(self.flows[0]); self.write()
        with self.assertRaisesRegex(ValueError, "duplicate flow"):
            eda.load_inputs(self.root)

    def test_duplicate_or_missing_feature_rejected(self):
        self.features.append(self.features[0]); self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate feature"):
            eda.load_inputs(self.root)
        self.features = self.features[1:-1]; self.write()
        with self.assertRaisesRegex(ValueError, "feature representation"):
            eda.load_inputs(self.root)

    def test_unknown_label_and_metadata_mismatch_rejected(self):
        self.features[0]["candidate_class"] = "VOIP"; self.write()
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            eda.load_inputs(self.root)
        self.features[0]["candidate_class"] = "invented"; self.write()
        with self.assertRaisesRegex(ValueError, "Unexpected candidate"):
            eda.load_inputs(self.root)

    def test_delay_units_and_common_cohort(self):
        tables = self.tables()
        common = tables["delay_common_n20_cohort"]
        self.assertTrue((common["count"] == 1).all())
        self.assertEqual(common.loc[common.n_packets == 20, "median_ms"].iloc[0], 20)

    def test_complete_pipeline_preserves_source_bytes(self):
        before = {p: p.read_bytes() for p in self.folder.iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            eda.run(self.root)
        self.assertEqual(before, {p: p.read_bytes() for p in self.folder.iterdir()})
        self.assertEqual(len(list((self.root / "results/eda/plots").glob("*.png"))), 12)
        self.assertTrue((self.root / "results/eda/REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
