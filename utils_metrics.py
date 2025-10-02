import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

def relative_reconstruction_error(V_clean, W, H):
    """
    RRE = ||V_clean - W H||_F / ||V_clean||_F
    """
    num = np.linalg.norm(V_clean - W @ H)
    den = np.linalg.norm(V_clean) + 1e-12
    return num / den

def assign_cluster_label(H, y_true):
    """
    KMeans on code matrix (samples × k). 
    Maps each cluster to the majority label among its members.
    Returns predicted labels aligned to y_true's indexing.
    """
    n_clusters = len(set(y_true))
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit(H)
    labels = km.labels_

    y_pred = np.zeros_like(y_true)
    for c in range(n_clusters):
        idx = (labels == c)
        if np.any(idx):
            majority = Counter(y_true[idx]).most_common(1)[0][0]
            y_pred[idx] = majority
    return y_pred

def evaluate(V_clean, W, H, y_true):
    """
    Convenience: compute RRE, Accuracy, NMI.
    - Clustering is done on H.T (samples × k) as in the assignment example.
    """
    rre = relative_reconstruction_error(V_clean, W, H)
    y_pred = assign_cluster_label(H.T, y_true)
    acc = accuracy_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return rre, acc, nmi