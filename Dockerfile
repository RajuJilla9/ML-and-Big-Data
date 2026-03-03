# ============================================================
# Big Data ML Project - Production Dockerfile
# PySpark + MLlib + Scikit-Learn + Testing Support
# ============================================================

# ----------------------------
# Base Image
# ----------------------------
FROM python:3.11-slim

# ----------------------------
# Install System Dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    wget \
    curl \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Java Environment
# ----------------------------
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ----------------------------
# Install Apache Spark
# ----------------------------
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3

RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# ----------------------------
# Set Working Directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Copy requirements first (for Docker cache efficiency)
# ----------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Copy Project Files
# ----------------------------
COPY . .

# ----------------------------
# Create Required Directories
# ----------------------------
RUN mkdir -p \
    data/raw \
    data/parquet \
    models \
    outputs \
    checkpoints \
    logs

# ----------------------------
# Expose Spark UI
# ----------------------------
EXPOSE 4040

# ----------------------------
# Default Command
# ----------------------------
CMD ["python", "scripts/run_pipeline.py"]