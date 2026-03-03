#!/usr/bin/env bash
# ============================================================
# setup_environment.sh
# Big Data ML Environment Setup Script
# ============================================================

set -e

echo "=================================================="
echo " Setting up Big Data ML Environment"
echo "=================================================="

# ----------------------------
# 1. Python Virtual Environment
# ----------------------------
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required packages..."
pip install \
    pyspark==3.5.1 \
    numpy \
    pandas \
    pyarrow \
    scikit-learn \
    matplotlib \
    seaborn \
    wordcloud \
    findspark \
    tqdm

echo "Python dependencies installed."

# ----------------------------
# 2. Spark Environment
# ----------------------------
if [ -z "$SPARK_HOME" ]; then
    echo "SPARK_HOME not set. Please install Apache Spark and export SPARK_HOME."
else
    echo "SPARK_HOME detected at: $SPARK_HOME"
fi

# ----------------------------
# 3. Create project directories
# ----------------------------
mkdir -p data/raw
mkdir -p data/parquet
mkdir -p models
mkdir -p outputs
mkdir -p checkpoints
mkdir -p logs

echo "Project directories created."

# ----------------------------
# 4. Spark Configuration Template
# ----------------------------
cat <<EOF > spark-defaults.conf

spark.executor.memory              4g
spark.executor.cores               4
spark.driver.memory                4g
spark.sql.shuffle.partitions       200
spark.sql.adaptive.enabled         true
spark.serializer                   org.apache.spark.serializer.KryoSerializer
spark.sql.execution.arrow.enabled  true

EOF

echo "Spark config template written to spark-defaults.conf"

echo "=================================================="
echo " Environment Setup Complete"
echo "=================================================="