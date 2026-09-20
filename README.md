# Clustering: k-Means, Hierarchical, GMM & DBSCAN

A four-part Jupyter notebook series that explains unsupervised clustering from the ground up — starting with the core intuition and k-Means, then hierarchical clustering, Gaussian Mixture Models, and finally DBSCAN — with `scikit-learn` code walkthroughs on real customer datasets.

## Contents

| # | Notebook | Topics |
|---|---|---|
| 1 | 1__Intro_to_Clustering__k_Means.ipynb | Supervised vs. unsupervised learning; what clustering is; intra-cluster vs. inter-cluster distance; distance metrics (Euclidean, Manhattan, etc.); how to judge a clustering; **k-Means** (centroid-based clustering, Lloyd's algorithm, choosing *K* via WCSS/elbow method); an **RFM (Recency, Frequency, Monetary) customer segmentation** case study on an online retail transaction dataset, including data cleaning, outlier treatment, scaling, k-Means fitting, Silhouette Analysis, and elbow-method cluster selection with business interpretation of the resulting segments |
| 2 | 2__K_means__Heirarchical.ipynb | Limitations of k-Means (initialization sensitivity, poor handling of varying cluster sizes/densities/shapes); **k-Means++** initialization; **Hierarchical clustering** — Agglomerative (bottom-up) vs. Divisive (top-down); the agglomerative algorithm (proximity matrix, iterative merging); linkage methods (single, complete, Ward); dendrograms; a **customer spending segmentation** case study using `AgglomerativeClustering` with average/complete/ward linkage dendrograms and cluster-count selection; advantages/disadvantages of hierarchical clustering and a k-Means vs. agglomerative comparison |
| 3 | 3__GMM.ipynb | **Gaussian Mixture Models (GMM)** via `sklearn.mixture.GaussianMixture` — fitting soft/probabilistic cluster assignments (`predict_proba`), component weights, means, and covariances — on a PCA-reduced e-commerce dataset; visualizing clusters in both PCA and **t-SNE** space; a direct **GMM vs. k-Means** comparison, including a synthetic multivariate-Gaussian dataset built to highlight how the two algorithms differ on overlapping/non-spherical clusters |
| 4 | 4__DBSCAN.ipynb | **DBSCAN** (density-based clustering) via `sklearn.cluster.DBSCAN` on the same PCA-reduced e-commerce dataset — fitting with `eps`/`min_samples`, inspecting resulting cluster labels (including noise points), and visualizing clusters in PCA space; evaluating clustering quality with the **Davies–Bouldin score** and **Silhouette score** |

## Suggested learning path

The notebooks build on one another and are meant to be worked through in order:

1. **`1__Intro_to_Clustering__k_Means.ipynb`** — clustering fundamentals and k-Means, applied to an RFM customer-segmentation case study.
2. **`2__K_means__Heirarchical.ipynb`** — where k-Means breaks down, k-Means++, and hierarchical clustering as an alternative, applied to a customer-spending case study.
3. **`3__GMM.ipynb`** — soft, probabilistic clustering with Gaussian Mixture Models and how it differs from k-Means.
4. **`4__DBSCAN.ipynb`** — density-based clustering with DBSCAN and quantitative cluster-quality metrics.

## Requirements

- Python 3
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `gdown` and/or internet access (`wget`) — used to download the datasets in notebooks 1, 2, 3, and 4
- `IPython` (for inline images/display in notebook 2)

Install with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn gdown ipython
```

## Datasets

- **Online Retail transactional dataset** (`Online_Retail.csv`, notebook 1) — downloaded via `gdown`; used to engineer Recency/Frequency/Monetary features for k-Means-based customer segmentation.
- **Customer spending dataset** (`Cust_Spend_Data.csv`, notebook 2) — downloaded via `gdown`; used for the agglomerative/hierarchical clustering case study.
- **E-commerce dataset** (`E-commerce.csv`, notebooks 3 and 4) — downloaded via `wget` from Google Drive; standardized and PCA-reduced before fitting GMM and DBSCAN respectively.

All datasets are fetched at runtime within their respective notebooks; no separate data directory is required.

## How to run

Open each notebook with Jupyter or JupyterLab (or Google Colab, given the `gdown`/`wget` download cells) and run all cells top to bottom:

```bash
jupyter notebook 1__Intro_to_Clustering__k_Means.ipynb
```

Cells within each notebook are sequential — later sections depend on the data-loading, cleaning, and scaling steps earlier in the same notebook — so run them in order. The four notebooks are otherwise independent of one another (each downloads and prepares its own data).

## Notes

- These notebooks are instructional: notebooks 1 and 2 include instructor notes, in-class questions, and quizzes alongside the code, and are best suited for guided teaching or self-paced learning rather than as a production clustering library. Notebooks 3 and 4 are lighter on prose and lean more heavily on the code itself.
