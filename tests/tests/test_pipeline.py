"""
test_pipeline.py
============================================================
Unit & Integration Tests for Big Data ML Pipeline
Run with:

    pytest tests/

============================================================
"""

import os
import json
import pytest
import yaml
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeansModel, BisectingKMeansModel


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder \
        .appName("PipelineTestSession") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def config():
    config_path = "config/spark_config.yaml"
    assert os.path.exists(config_path), "Missing spark_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# 1. Config Validation
# ============================================================

def test_config_structure(config):
    assert "spark" in config
    assert "ml" in config
    assert "data" in config
    assert "evaluation" in config


def test_spark_config_values(config):
    assert config["spark"]["resources"]["driver_memory"] is not None
    assert config["spark"]["sql"]["shuffle_partitions"] > 0


# ============================================================
# 2. Data Integrity Tests
# ============================================================

def test_parquet_files_exist():
    required_files = [
        "data/parquet/train_features.parquet",
        "data/parquet/val_features.parquet",
        "data/parquet/test_features.parquet",
        "data/parquet/tfidf_full.parquet",
    ]
    for f in required_files:
        assert os.path.exists(f), f"Missing file: {f}"


def test_feature_schema(spark):
    df = spark.read.parquet("data/parquet/train_features.parquet")
    assert "docID" in df.columns
    assert "pca_50_features" in df.columns
    assert df.count() > 0


# ============================================================
# 3. Model Existence Tests
# ============================================================

def test_models_saved():
    model_paths = [
        "models/kmeans_best",
        "models/bisecting_kmeans_best",
        "models/lda_best",
        "models/gmm_final.pkl",
    ]
    for m in model_paths:
        assert os.path.exists(m), f"Model not found: {m}"


def test_kmeans_model_loadable(spark):
    model = KMeansModel.load("models/kmeans_best")
    assert model.getK() > 0


def test_bisecting_model_loadable(spark):
    model = BisectingKMeansModel.load("models/bisecting_kmeans_best")
    assert model.getK() > 0


# ============================================================
# 4. Evaluation Output Tests
# ============================================================

def test_evaluation_metrics_exist():
    path = "outputs/evaluation_metrics.csv"
    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert not df.empty
    assert "Silhouette (sklearn)" in df.columns


def test_scalability_files_exist():
    assert os.path.exists("outputs/strong_scaling.csv")
    assert os.path.exists("outputs/weak_scaling.csv")


def test_training_results_json():
    path = "outputs/training_results.json"
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "KMeans" in data


# ============================================================
# 5. Sanity Clustering Test
# ============================================================

def test_cluster_predictions_valid(spark):
    df = spark.read.parquet("data/parquet/test_features.parquet")
    model = KMeansModel.load("models/kmeans_best")
    preds = model.transform(df)
    assert "cluster" in preds.columns
    distinct_clusters = preds.select("cluster").distinct().count()
    assert distinct_clusters > 1


# ============================================================
# 6. Performance Profiler Output Test
# ============================================================

def test_performance_profile_exists():
    path = "outputs/performance_profile.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        assert "strong_scaling" in data


# ============================================================
# END OF TESTS
# ============================================================