"""
run_pipeline.py
============================================================
Runs full Big Data ML pipeline:
1. Ingestion
2. Feature Engineering
3. Model Training
4. Evaluation
============================================================
"""

import os
import subprocess
import sys
import time
import logging

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

NOTEBOOKS = [
    "notebooks/1_data_ingestion.ipynb",
    "notebooks/2_feature_engineering.ipynb",
    "notebooks/3_model_training.ipynb",
    "notebooks/4_evaluation.ipynb",
]

def run_notebook(nb_path):
    print(f"\nRunning: {nb_path}")
    logging.info(f"Starting {nb_path}")
    start = time.time()

    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        nb_path,
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        logging.error(f"Notebook failed: {nb_path}")
        raise RuntimeError(f"Execution failed for {nb_path}")

    elapsed = time.time() - start
    logging.info(f"Completed {nb_path} in {elapsed:.2f}s")
    print(f"Completed in {elapsed:.2f}s")


def main():
    print("==================================================")
    print(" Running Full Big Data ML Pipeline")
    print("==================================================")

    total_start = time.time()

    for nb in NOTEBOOKS:
        run_notebook(nb)

    total_time = time.time() - total_start
    print("\nPipeline completed successfully.")
    print(f"Total execution time: {total_time:.2f}s")
    logging.info(f"Pipeline finished in {total_time:.2f}s")


if __name__ == "__main__":
    main()