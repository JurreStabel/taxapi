from pathlib import Path

# taxapi/core/paths.py -> parents[2] is the repo root
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BENCHMARK_DIR = DATA_DIR / "benchmark"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"


def benchmark_dir(relaxed=False):
    return BENCHMARK_DIR / "relaxed" if relaxed else BENCHMARK_DIR
