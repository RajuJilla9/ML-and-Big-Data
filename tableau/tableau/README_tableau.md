# Tableau Dashboard Documentation

## Big Data & Machine Learning -- Clustering Analysis

This Tableau workbook presents interactive visual analytics for
clustering experiments conducted using PySpark MLlib and scikit-learn on
the Bag-of-Words (KOS) dataset.

The dashboards support:

-   Data quality analysis
-   Model performance comparison
-   Cluster structure interpretation
-   Scalability & computational efficiency evaluation

All datasets visualized in Tableau were exported from Spark processing
pipelines.

------------------------------------------------------------------------

# Dashboard Overview

------------------------------------------------------------------------

## 📊 DASHBOARD 1 --- Data Quality & EDA

Dataset used: `doc_lengths.csv`

Purpose:\
To analyze document structure, distribution characteristics, and lexical
complexity before model training.

### Visualizations Included:

1.  Unique Terms Distribution (Histogram)\
2.  Document Length Distribution (Histogram)\
3.  Complexity Relationship (Scatter Plot with Trend Line)\
4.  Document Complexity Ratio (Histogram of Vocabulary Density)

Key Insight:\
The dataset exhibits long-tail distributions and strong correlation
between document length and vocabulary growth, consistent with Zipf-like
properties in text corpora.

------------------------------------------------------------------------

## 📊 DASHBOARD 2 --- Model Performance Comparison

Dataset used: `evaluation_metrics.csv`

Algorithms Compared: - KMeans - BisectingKMeans - GMM

### Visualizations Included:

1.  Silhouette Score Comparison (Spark & sklearn)\
2.  Davies-Bouldin Index Comparison\
3.  Calinski-Harabasz Score Comparison\
4.  Separation vs Compactness Scatter Plot

Key Insight:\
Models positioned in the upper-left region (high silhouette, low DB
index) demonstrate superior structural clustering quality.

------------------------------------------------------------------------

## 📊 DASHBOARD 3 --- Clustering Structure

Dataset used: `tsne_clusters.csv`

### Interactive Feature

Parameter: Select Algorithm\
Options: - KMeans - BisectingKMeans - GMM

Calculated Field:

CASE \[Select Algorithm\]\
WHEN "KMeans" THEN \[km_cluster\]\
WHEN "BisectingKMeans" THEN \[bkm_cluster\]\
WHEN "GMM" THEN \[gmm_cluster\]\
END

### Visualizations Included:

1.  t-SNE Scatter Plot (colored by selected cluster)\
2.  Cluster Count Chart (COUNT(docID) by cluster)\
3.  Measure Values vs Selected Cluster\
4.  Dynamic Algorithm Selection Control

Key Insight:\
Cluster separation and distribution patterns vary across algorithms,
highlighting differences in structural modeling approaches.

------------------------------------------------------------------------

## 📊 DASHBOARD 4 --- Scalability & Performance

Dataset used: `scalability_results.csv`

### Visualizations Included:

1.  Training Time vs Dataset Size (Weak Scaling)\
2.  Training Time vs Cores (Strong Scaling)\
3.  Algorithm Cores vs Training Time (Scatter Plot)\
4.  Comparative Scaling Interpretation

Key Insight:\
Training time trends demonstrate distributed scalability behavior and
reveal cost-performance trade-offs in parallel machine learning
workflows.

------------------------------------------------------------------------

# Tableau Techniques Used

-   Parameter Controls for dynamic algorithm switching\
-   Calculated Fields for cluster mapping\
-   Trend Lines for correlation analysis\
-   Histogram binning for distribution study\
-   Scatter plots for multi-metric evaluation\
-   Extract (.hyper) data optimization

------------------------------------------------------------------------

# Data Processing Workflow

1.  Data ingestion via Spark from Bag-of-Words dataset\
2.  Feature engineering using TF-IDF and PCA\
3.  Model training (KMeans, BisectingKMeans, GMM)\
4.  Evaluation metric computation\
5.  t-SNE dimensionality reduction\
6.  Export to CSV and visualization in Tableau

------------------------------------------------------------------------

# Reproducibility

To regenerate dashboards:

1.  Run notebooks in order:

    -   Data Ingestion\
    -   Feature Engineering\
    -   Model Training\
    -   Evaluation

2.  Export required CSV files from `/outputs`.

3.  Open Tableau workbook and refresh data sources.

------------------------------------------------------------------------

# Conclusion

These dashboards provide a complete analytical visualization layer for
distributed clustering experiments. They support interpretability,
performance comparison, structural validation, and scalability
evaluation within a big data machine learning context.
