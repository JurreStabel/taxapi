import logging
import time

from tqdm import tqdm

_LOGGER_NAME = "taxapi"


def get_logger():
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def run_pipeline(name, steps):
    # steps: list of (label, callable). Runs each in order with logging + a tqdm bar.
    log = get_logger()
    log.info("starting %s pipeline (%d steps)", name, len(steps))
    timings = []
    for label, fn in tqdm(steps, desc=name, unit="step"):
        log.info("step: %s", label)
        start = time.perf_counter()
        try:
            fn()
        except Exception:
            log.exception("step failed: %s", label)
            raise
        elapsed = time.perf_counter() - start
        log.info("done:  %s (%.1fs)", label, elapsed)
        timings.append((label, elapsed))
    total = sum(t for _, t in timings)
    log.info("%s pipeline finished: %d steps in %.1fs", name, len(timings), total)
    return timings
