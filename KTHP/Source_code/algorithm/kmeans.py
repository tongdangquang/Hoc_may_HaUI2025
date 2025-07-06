import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from algorithm import *

def partition_coefficient(membership_matrix):
    return np.sum(membership_matrix ** 2) / membership_matrix.shape[0]

def euclidean(a, b):
    return np.linalg.norm(a - b)

def dunn_index(data, labels, centers):
    n_clusters = len(np.unique(labels))
    inter_cluster = [
        euclidean(centers[i], centers[j])
        for i in range(n_clusters) for j in range(i+1, n_clusters)
    ]
    min_inter = np.min(inter_cluster) if inter_cluster else 0

    max_intra = 0
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 1:
            dists = [euclidean(p1, p2) for p1 in cluster_points for p2 in cluster_points]
            max_intra = max(max_intra, max(dists))
    return min_inter / max_intra if max_intra != 0 else 0

def kmeans_algo(data, num_clusters=3, max_iter=300):
    if isinstance(data, pd.DataFrame):
        data = data.values

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    start = time.time()
    kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    end = time.time()

    # Membership matrix (1-hot)
    n_samples = len(data)
    membership = np.zeros((n_samples, num_clusters))
    membership[np.arange(n_samples), labels] = 1

    # Tính chỉ số
    db = round(davies_bouldin_score(scaled_data, labels), 5)
    pc = round(partition_coefficient(membership), 5)
    di = round(dunn_index(scaled_data, labels, kmeans.cluster_centers_), 5)
    exec_time = round(end - start, 5)
    iters = kmeans.n_iter_

    result_df = pd.DataFrame([['KMEANS', iters, exec_time, db, pc, di]],
                             columns=["Algo", "Iters", "Time", "DB", "PC", "DI"])
    return result_df

tmp=pd.read_csv("Dry_Bean_Dataset_cleaned.csv")
data, target=np.array(tmp.iloc[:, 0:15]), pd.factorize(np.array(tmp.iloc[:,15]))[0]
kmeans = kmeans_algo(data, num_clusters=7, max_iter=1000)
print(kmeans)
