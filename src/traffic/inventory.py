
# A1 — Dataset Inventory
# Identify available PCAP/PCAPNG files and record their metadata.
# Verify dataset availability before processing.

import csv

from config import DATA_FOLDERS, RESULTS_DIR


OUTPUT = RESULTS_DIR / "dataset_inventory.csv"

INVENTORY_COLUMNS = [
    "capture_id",
    "filename",
    "format",
    "size_mb",
    "service_label",
    "session_id",
]


def discover_captures():
    captures = []

    for folder in DATA_FOLDERS:
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}:
                capture_id = f"{folder.name}/{path.relative_to(folder).as_posix()}"
                captures.append((path, capture_id))

    return captures


def build_inventory():
    records = []

    for path, capture_id in discover_captures():
        records.append({
            "capture_id": capture_id,
            "filename": path.name,
            "format": path.suffix.lower(),
            "size_mb": round(path.stat().st_size / (1024 ** 2), 2),
            "service_label": "",
            "session_id": "",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    print(f"Found {len(records)} captures")
    print(f"Saved: {OUTPUT}")

    return records


if __name__ == "__main__":
    build_inventory()
