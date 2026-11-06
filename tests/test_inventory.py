
# Test A1 — Dataset Inventory
# Verify discovery of PCAP and PCAPNG files across configured folders.
# Test inventory metadata and missing-folder handling using temporary data.

import csv

import pytest

from src.traffic import inventory


def test_discover_captures(tmp_path, monkeypatch):
    folder1 = tmp_path / "NonVPN-PCAPs-01"
    folder2 = tmp_path / "NonVPN-PCAPs-02"
    folder3 = tmp_path / "NonVPN-PCAPs-03"

    for folder in (folder1, folder2, folder3):
        folder.mkdir()

    (folder1 / "aim_chat.pcap").write_bytes(b"abc")
    (folder1 / "ignore.txt").write_text("not a capture")
    (folder2 / "facebook_audio.pcapng").write_bytes(b"abcdef")

    nested = folder3 / "nested"
    nested.mkdir()
    (nested / "netflix.PCAP").write_bytes(b"123456789")

    monkeypatch.setattr(
        inventory,
        "DATA_FOLDERS",
        [folder1, folder2, folder3],
    )

    captures = inventory.discover_captures()

    capture_ids = {capture_id for _, capture_id in captures}

    assert len(captures) == 3
    assert capture_ids == {
        "NonVPN-PCAPs-01/aim_chat.pcap",
        "NonVPN-PCAPs-02/facebook_audio.pcapng",
        "NonVPN-PCAPs-03/nested/netflix.PCAP",
    }

    assert all(path.is_file() for path, _ in captures)


def test_missing_dataset_folder(tmp_path, monkeypatch):
    existing = tmp_path / "NonVPN-PCAPs-01"
    missing = tmp_path / "NonVPN-PCAPs-02"

    existing.mkdir()

    monkeypatch.setattr(
        inventory,
        "DATA_FOLDERS",
        [existing, missing],
    )

    with pytest.raises(FileNotFoundError, match="Dataset folder not found"):
        inventory.discover_captures()


def test_build_inventory(tmp_path, monkeypatch):
    folder = tmp_path / "NonVPN-PCAPs-01"
    folder.mkdir()

    (folder / "netflix1.pcap").write_bytes(b"x" * 2048)

    output = tmp_path / "results" / "dataset_inventory.csv"

    monkeypatch.setattr(inventory, "DATA_FOLDERS", [folder])
    monkeypatch.setattr(inventory, "OUTPUT", output)

    records = inventory.build_inventory()

    assert len(records) == 1

    record = records[0]

    assert record["capture_id"] == "NonVPN-PCAPs-01/netflix1.pcap"
    assert record["filename"] == "netflix1.pcap"
    assert record["format"] == ".pcap"
    assert record["size_mb"] == round(2048 / (1024 ** 2), 2)
    assert record["service_label"] == ""
    assert record["session_id"] == ""

    assert output.is_file()

    with output.open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))

    assert len(saved) == 1
    assert saved[0]["capture_id"] == record["capture_id"]
    assert float(saved[0]["size_mb"]) == record["size_mb"]
    assert saved[0]["service_label"] == ""
    assert saved[0]["session_id"] == ""
