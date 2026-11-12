"""Run with unittest as well as pytest; no raw data or third-party packages needed."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.traffic import label_captures as labels


class FilenameRulesTest(unittest.TestCase):
    def test_documented_candidates(self):
        cases = {"AIMchat1.pcapng": "CHAT", "aim_chat_3a.pcap": "CHAT",
                 "ICQchat2.pcapng": "CHAT", "icq_chat_3b.pcap": "CHAT",
                 "facebookchat3.pcapng": "CHAT", "facebook_chat_4a.pcap": "CHAT",
                 "hangout_chat_4b.pcap": "CHAT", "hangouts_chat_4a.pcap": "CHAT",
                 "email2b.pcap": "MAIL", "facebook_audio1b.pcapng": "VOIP",
                 "hangouts_audio2a.pcap": "VOIP", "ftps_down_1a.pcap": "FT"}
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                row = labels.infer_capture("NonVPN-PCAPs-01/" + filename)
                self.assertEqual(row["candidate_class"], expected)
                self.assertEqual(row["verification_status"], "capture_derived_unverified")
                self.assertIsNone(row["recording_group"])

    def test_ambiguous_and_unknown_fail_closed(self):
        for name in ("facebook_video1a.pcap", "hangouts_video2b.pcapng",
                     "netflix1.pcap", "gmailchat1.pcapng", "unknown_chat1.pcap",
                     "facebook_audio1a_extra.pcap", "email1a.txt", "vpn_aim_chat1.pcap"):
            with self.subTest(name=name):
                self.assertEqual(labels.infer_capture(name)["candidate_class"], "UNRESOLVED")

    def test_case_and_path(self):
        self.assertEqual(labels.infer_capture(r"folder\FACEBOOKCHAT1.PCAPNG")["candidate_class"], "CHAT")


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cid = "NonVPN-PCAPs-01/AIMchat1.pcapng"
        self.inventory = [{"capture_id": self.cid, "filename": "AIMchat1.pcapng"}]
        self.flows = [{"flow_id": "f1", "capture_id": self.cid, "protocol": "TCP",
                       "packet_count": "20", **{f"eligible_n{n}": "1" for n in labels.BUDGETS}}]
        self.features = [{"flow_id": "f1", "capture_id": self.cid, "protocol": "TCP",
                          "n_packets": str(n), "mean_length": "62.00000000000001"}
                         for n in labels.BUDGETS]
        for path, rows in (("results/dataset_inventory.csv", self.inventory),
                           ("data/processed/all_flows.csv", self.flows),
                           ("data/processed/all_features.csv", self.features)):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            labels.write_csv(target, list(rows[0]), rows)

    def test_join_preserves_every_input_value_and_order(self):
        manifest = labels.build_manifest(self.inventory)
        columns = list(self.features[0])
        out_cols, joined = labels.join_metadata(columns, self.features, manifest)
        self.assertEqual(out_cols[:len(columns)], columns)
        self.assertEqual([{k: r[k] for k in columns} for r in joined], self.features)

    def test_join_rejects_unmapped_and_duplicate_manifest(self):
        manifest = labels.build_manifest(self.inventory)
        with self.assertRaises(ValueError):
            labels.join_metadata(list(self.features[0]), self.features, [])
        with self.assertRaises(ValueError):
            labels.join_metadata(list(self.features[0]), self.features, manifest * 2)

    def test_inventory_rejects_duplicate_and_filename_mismatch(self):
        with self.assertRaises(ValueError):
            labels.build_manifest(self.inventory * 2)
        with self.assertRaises(ValueError):
            labels.build_manifest([dict(self.inventory[0], filename="other.pcap")])

    def test_alignment_rejects_missing_duplicate_and_unknown_features(self):
        manifest = labels.build_manifest(self.inventory)
        for rows in (self.features[:-1], self.features + self.features[:1],
                     [dict(self.features[0], flow_id="unknown"), *self.features[1:]],
                     [dict(self.features[0], capture_id="wrong"), *self.features[1:]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                labels.validate_alignment(self.flows, rows, manifest)

    def test_full_run_reproducible_and_originals_unchanged(self):
        originals = {p: p.read_bytes() for p in self.root.rglob("*.csv")}
        report = labels.run(self.root)
        first = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        labels.run(self.root)
        self.assertEqual(first, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertTrue(all(p.read_bytes() == content for p, content in originals.items()))
        self.assertEqual(report["verified_flow_labels"], 0)
        manifest = json.loads((self.root / "results/labeling/capture_manifest.json").read_text())
        self.assertIsNone(manifest[0]["recording_group"])
        _, output = labels.read_table(self.root / "data/processed/capture_labeled/all_features_capture_labeled.csv")
        self.assertEqual([{k: row[k] for k in self.features[0]} for row in output], self.features)

    def test_summary_keeps_missing_classes_and_unknown_groups(self):
        summary = labels.eligibility_summary(self.flows, labels.build_manifest(self.inventory), 100)
        self.assertEqual(len(summary), 32)
        chat = next(r for r in summary if r["candidate_class"] == "CHAT" and r["n_packets"] == 20)
        self.assertEqual(chat["eligible_flows"], 1)
        self.assertEqual(chat["documented_recording_groups"], 0)
        self.assertEqual(chat["group_split_status"], "BLOCKED_unknown_independence")
        self.assertEqual(chat["n20_support_status"], "below_screening_threshold")


if __name__ == "__main__":
    unittest.main()
