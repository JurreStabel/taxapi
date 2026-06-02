# Build the benchmark datasets end-to-end: raw -> clean -> +WOZ -> +source 4.
# Run: python -m taxapi.pipelines.train             (gold / strict)
#      python -m taxapi.pipelines.train --relaxed    (relaxed +30d -> data/benchmark/relaxed/)
import sys

from taxapi.builders import clean, source4, woz
from taxapi.core import runner


def main():
    relaxed = "--relaxed" in sys.argv
    label = "train (relaxed)" if relaxed else "train"
    steps = [
        ("build clean benchmark", lambda: clean.main(relaxed=relaxed)),
        ("add WOZ", lambda: woz.main(relaxed=relaxed)),
        ("add source 4", lambda: source4.main(relaxed=relaxed)),
    ]
    runner.run_pipeline(label, steps)


if __name__ == "__main__":
    main()
