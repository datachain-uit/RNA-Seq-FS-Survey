import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FS_RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "fs")
DL_RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "dl")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

for path in [DATA_DIR, FS_RESULT_DIR, DL_RESULT_DIR, LOG_DIR]:
    os.makedirs(path, exist_ok=True)
