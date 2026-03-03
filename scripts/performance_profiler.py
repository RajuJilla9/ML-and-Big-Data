"""
performance_profiler.py
============================================================
Performance & Scalability Profiler for Big Data ML Project
============================================================
"""

import time
import psutil
import json
import os
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans

OUTPUT_FILE = "./outputs/performance_profile.json"
os.makedirs("./outputs", exist_ok=True)

def create_spark(partitions=4):
    spark = SparkSession.builder \
        .appName("PerformanceProfiler") \
        .master(f"local[{partitions}]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", partitions * 2) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


def monitor_resources():
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_mb": psutil.virtual_memory().used / (1024 ** 2),
    }


def strong_scaling_test(data_path):
    print("Running Strong Scaling Test...")
    results = []

    for cores in [1, 2, 4, 8]:
        spark = create_spark(cores)
        df = spark.read.parquet(data_path)
        df.cache()
        df.count()

        km = KMeans(
            featuresCol="pca_50_features",
            predictionCol="cluster",
            k=10,
            maxIter=20,
            seed=42
        )

        start = time.time()
        km.fit(df)
        elapsed = time.time() - start

        resources = monitor_resources()

        results.append({
            "cores": cores,
            "execution_time_s": elapsed,
            "resources": resources
        })

        spark.stop()

    return results


def main():
    DATA_PATH = "./data/parquet/train_features.parquet"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("train_features.parquet not found.")

    scaling_results = strong_scaling_test(DATA_PATH)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strong_scaling": scaling_results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("\nPerformance profiling complete.")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()