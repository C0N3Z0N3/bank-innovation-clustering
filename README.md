# Bank Innovation Clustering and Prediction

**University of Georgia | Data Science Capstone | 2026**

## Overview

An unsupervised machine learning pipeline that identifies innovation behavioral patterns across U.S. commercial banks using FDIC Call Report data. The model clusters banks by financial change behavior and predicts cluster migration using deep learning, achieving 72.7% cross-validated accuracy.

## Project Structure

## What This Project Does

- Builds 3-year rolling financial change scores across 20 proxy ratios for approximately 8,000 U.S. commercial banks (2010-2021)
- Applies UMAP dimensionality reduction and HDBSCAN clustering to identify 4-5 distinct innovation behavioral clusters per asset tier
- Tracks 5,253 cluster migration events across 2,623 banks over time
- Develops GRU and LSTM neural networks in TensorFlow trained on 42,000+ observations to predict future cluster membership
- Cluster trajectories independently recovered COVID-19 and pre-COVID branch expansion patterns without using any time-based features

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, umap-learn, hdbscan, TensorFlow, matplotlib, seaborn
- **Models:** UMAP, HDBSCAN, GRU, LSTM
- **Environment:** JupyterLab

## Data

Raw data sourced from FDIC Call Reports, publicly available through the FDIC bulk download portal:
https://www.fdic.gov/bank/statistical/guide/2023/index.html

Note: CSV data files have been removed from this repository due to file size limitations and confidentiality considerations. To reproduce this analysis, download the relevant Call Report files from the FDIC portal and place them in the `/data` directory before running the notebooks.

## Results

- 72.7% cross-validated accuracy on cluster migration prediction
- 4-5 behavioral clusters identified per bank asset tier
- 5,253 cluster migration events tracked across 2,623 banks
- COVID-19 period recovered naturally through unsupervised clustering without temporal features

## Author

**Connor Ray**
University of Georgia, B.S. Data Science, Class of 2026
[LinkedIn](https://www.linkedin.com/in/connor-ray-4ba496308/) | [GitHub](https://github.com/C0N3Z0N3)
