import logging
import os
from datetime import datetime

log_dir = "test/logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"test_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="a", encoding="utf-8")],
)

logging.info(f"Logging initialized. Log file: {log_filename}")
