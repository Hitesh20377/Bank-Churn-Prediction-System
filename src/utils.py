import logging
import os

def setup_logger():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    log_path = os.path.join(BASE_DIR, "logs", "pipeline.log")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )