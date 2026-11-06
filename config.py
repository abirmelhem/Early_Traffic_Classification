
# Shared Project Configuration
# Define dataset locations, output folders, and experiment settings.
# Keep settings consistent across all scripts and computers.

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

DATA_ROOT = Path(
    os.getenv("TRAFFIC_DATA_ROOT", str(Path.home() / "Downloads"))
)

DATA_FOLDERS = [
    DATA_ROOT / "NonVPN-PCAPs-01",
    DATA_ROOT / "NonVPN-PCAPs-02",
]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

PACKET_BUDGETS = [3, 5, 10, 20]
MAX_PREFIX = max(PACKET_BUDGETS)

TIMEOUT = 60.0
RANDOM_STATE = 42
