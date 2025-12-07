"""
XRD Analysis Module
NMF decomposition, clustering, and dimension reduction functions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from sklearn.decomposition import NMF
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

try:
    import dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def xrd_to_matrix(xrd_data: Dict[int, List]) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Convert XRD data dictionary to matrix format

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values

    Returns:
        Tuple of (intensity_matrix, two_theta, sample_ids)
    """
    keys = sorted(xrd_data.keys())
    values = [xrd_data[k] for k in keys]

    # Get 2theta from first sample
    two_theta = np.array(values[0][0])

    # Create intensity matrix
    y_matrix = np.array([v[1] for v in values])

    return y_matrix, two_theta, keys


def run_nmf(xrd_data: Dict[int, List],
            n_components: int = 10,
            max_iter: int = 1000,
            random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform NMF decomposition on XRD data

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values
        n_components: Number of basis vectors
        max_iter: Maximum iterations
        random_state: Random seed

    Returns:
        Tuple of (basis_vectors, coefficients, reconstruction_error_percent)
    """
    y_matrix, _, _ = xrd_to_matrix(xrd_data)

    # Run NMF
    nmf = NMF(
        n_components=n_components,
        max_iter=max_iter,
        init="nndsvd",
        random_state=random_state
    )
    nmf.fit(y_matrix)

    coefficient = nmf.fit_transform(y_matrix)  # (n_samples, n_components)
    basis_vector = nmf.components_  # (n_components, n_2theta)
    reconstruction_err = nmf.reconstruction_err_

    # Calculate reconstruction error percentage
    original_norm = np.linalg.norm(y_matrix, ord='fro')
    reconstruction_error_percent = (reconstruction_err / original_norm) * 100

    return basis_vector, coefficient, reconstruction_error_percent


def norm_array(arrays: np.ndarray, norm: float = 100) -> np.ndarray:
    """
    Normalize each array to a maximum value

    Args:
        arrays: 2D array (n_components, n_features)
        norm: Target maximum value

    Returns:
        Normalized array
    """
    new_arrays = np.zeros(arrays.shape)
    for i, one_vector in enumerate(arrays):
        max_intensity = np.max(one_vector)
        if max_intensity > 0:
            norm_vector = one_vector / max_intensity * norm
        else:
            norm_vector = one_vector
        new_arrays[i] = norm_vector
    return new_arrays


def calculate_dtw_matrix(arrays: np.ndarray,
                         window_size: int = 30,
                         show_progress: bool = True) -> np.ndarray:
    """
    Calculate DTW distance matrix between basis vectors

    Args:
        arrays: 2D array of basis vectors (n_components, n_features)
        window_size: Sakoe-Chiba window size
        show_progress: Whether to show progress bar

    Returns:
        Distance matrix (n_components, n_components)
    """
    if not HAS_DTW:
        raise ImportError("dtw-python is required for DTW distance calculation. "
                          "Install with: pip install dtw-python")

    n_data = len(arrays)
    dist = np.zeros([n_data, n_data])

    iterator = tqdm(range(n_data), desc="Calculating DTW") if show_progress else range(n_data)

    for i in iterator:
        for j in range(i, n_data):
            if i == j:
                dist[i][j] = 0
            else:
                alignment = dtw.dtw(
                    x=arrays[i],
                    y=arrays[j],
                    keep_internals=True,
                    window_type="sakoechiba",
                    window_args={"window_size": window_size}
                )
                d = alignment.distance
                dist[i][j] = d
                dist[j][i] = d

    return dist


def calculate_cosine_matrix(arrays: np.ndarray,
                            show_progress: bool = True) -> np.ndarray:
    """
    Calculate cosine distance matrix between basis vectors

    Args:
        arrays: 2D array of basis vectors (n_components, n_features)
        show_progress: Whether to show progress bar

    Returns:
        Distance matrix (n_components, n_components)
    """
    n_data = len(arrays)
    dist = np.zeros([n_data, n_data])

    iterator = tqdm(range(n_data), desc="Calculating cosine distance") if show_progress else range(n_data)

    for i in iterator:
        for j in range(i, n_data):
            if i == j:
                dist[i][j] = 0
            else:
                d = cosine(arrays[i], arrays[j])
                dist[i][j] = d
                dist[j][i] = d

    return dist


def calculate_correlation_matrix(arrays: np.ndarray,
                                 show_progress: bool = True) -> np.ndarray:
    """
    Calculate correlation-based distance matrix

    Args:
        arrays: 2D array of basis vectors
        show_progress: Whether to show progress bar

    Returns:
        Distance matrix
    """
    n_data = len(arrays)
    dist = np.zeros([n_data, n_data])

    iterator = tqdm(range(n_data), desc="Calculating correlation") if show_progress else range(n_data)

    for i in iterator:
        for j in range(i, n_data):
            if i == j:
                dist[i][j] = 0
            else:
                Pcorr, _ = pearsonr(arrays[i], arrays[j])
                pearson = (1 + Pcorr) / 2

                Scorr, _ = spearmanr(arrays[i], arrays[j])
                spearman = (1 + Scorr) / 2

                similarity = (pearson + spearman) / 2
                d = 1 - similarity

                dist[i][j] = d
                dist[j][i] = d

    return dist


def dim_reduction(dist_matrix: np.ndarray,
                  method: Literal["MDS", "tSNE", "UMAP"] = "MDS",
                  n_components: int = 2,
                  random_state: int = 1) -> np.ndarray:
    """
    Perform dimension reduction on distance matrix

    Args:
        dist_matrix: Precomputed distance matrix
        method: Dimension reduction method
        n_components: Number of output dimensions
        random_state: Random seed

    Returns:
        Embedding array (n_samples, n_components)
    """
    if method == "MDS":
        reducer = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state
        )
        embedding = reducer.fit_transform(dist_matrix)

    elif method == "tSNE":
        reducer = TSNE(
            n_components=n_components,
            perplexity=min(2, len(dist_matrix) - 1),
            init="random",
            random_state=random_state,
            metric="precomputed"
        )
        embedding = reducer.fit_transform(dist_matrix)

    elif method == "UMAP":
        if not HAS_UMAP:
            raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")

        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=min(2, len(dist_matrix) - 1),
            n_epochs=500,
            min_dist=1,
            init="spectral",
            metric="precomputed"
        )
        reducer.fit(dist_matrix)
        embedding = reducer.transform(dist_matrix)

    else:
        raise ValueError(f"Unknown method: {method}")

    return embedding


def run_dbscan(embedding: np.ndarray,
               eps: float = 20,
               min_samples: int = 1) -> np.ndarray:
    """
    Run DBSCAN clustering on embedding

    Args:
        embedding: 2D embedding array
        eps: Maximum distance between samples in a cluster
        min_samples: Minimum samples in a cluster

    Returns:
        Cluster labels array
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(embedding)
    return dbscan.labels_


def merge_cluster_ratio(coefficient: np.ndarray,
                        labels: np.ndarray) -> np.ndarray:
    """
    Merge NMF coefficients by cluster labels

    Args:
        coefficient: NMF coefficient matrix (n_samples, n_components)
        labels: Cluster labels for each component

    Returns:
        Merged cluster ratios (n_samples, n_clusters)
    """
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)
    n_samples = len(coefficient)

    cluster_ratio = np.zeros((n_samples, n_clusters))

    for cluster_idx, label in enumerate(unique_labels):
        indices = [j for j, x in enumerate(labels) if x == label]
        cluster_ratio[:, cluster_idx] = coefficient[:, indices].sum(axis=1)

    return cluster_ratio


def get_argmax_prob(cluster_ratio: np.ndarray) -> Tuple[List[int], List[List[float]]]:
    """
    Get argmax cluster and probability distribution for each sample

    Args:
        cluster_ratio: Cluster ratio matrix (n_samples, n_clusters)

    Returns:
        Tuple of (argmax_labels, probabilities)
    """
    argmax = []
    probabilities = []

    for row in cluster_ratio:
        row_sum = np.sum(row)
        if row_sum > 0:
            probability = (row / row_sum).tolist()
        else:
            probability = [1.0 / len(row)] * len(row)

        probabilities.append(probability)
        max_index = probability.index(max(probability))
        argmax.append(max_index)

    return argmax, probabilities


def normalize_coefficient_by_integral(basis_vector: np.ndarray,
                                      coefficient: np.ndarray) -> np.ndarray:
    """
    Normalize coefficients by integral of basis vectors

    Args:
        basis_vector: Basis vectors (n_components, n_features)
        coefficient: Coefficient matrix (n_samples, n_components)

    Returns:
        Normalized coefficient matrix
    """
    new_coefficient = coefficient.copy()

    for i in range(len(basis_vector)):
        y = basis_vector[i]
        x = np.arange(len(y))
        ymax = np.max(y)
        if ymax > 0:
            integral = np.trapz(y, x)
            new_coefficient[:, i] = coefficient[:, i] / ymax * integral

    return new_coefficient


def find_optimal_n_components(xrd_data: Dict[int, List],
                              max_components: int = 20,
                              step: int = 1) -> List[Tuple[int, float]]:
    """
    Find optimal number of NMF components by reconstruction error

    Args:
        xrd_data: XRD data dictionary
        max_components: Maximum number of components to test
        step: Step size for component search

    Returns:
        List of (n_components, error) tuples
    """
    results = []

    for n in range(2, max_components + 1, step):
        try:
            _, _, error = run_nmf(xrd_data, n_components=n)
            results.append((n, error))
        except Exception as e:
            print(f"Warning: NMF failed for n_components={n}: {e}")
            continue

    return results


def analyze_xrd_pipeline(xrd_data: Dict[int, List],
                         n_components: int = 10,
                         distance_method: str = "DTW",
                         dim_reduction_method: str = "MDS",
                         dbscan_eps: float = 20,
                         dbscan_min_samples: int = 1,
                         window_size: int = 30,
                         show_progress: bool = True) -> Dict:
    """
    Run full XRD analysis pipeline

    Args:
        xrd_data: Preprocessed XRD data dictionary
        n_components: Number of NMF components
        distance_method: Distance calculation method ("DTW", "cosine", "correlation")
        dim_reduction_method: Dimension reduction method ("MDS", "tSNE", "UMAP")
        dbscan_eps: DBSCAN eps parameter
        dbscan_min_samples: DBSCAN min_samples parameter
        window_size: DTW window size (only for DTW)
        show_progress: Whether to show progress

    Returns:
        Dictionary with analysis results
    """
    # Get 2theta
    _, two_theta, sample_ids = xrd_to_matrix(xrd_data)

    # Run NMF
    basis_vector, coefficient, recon_error = run_nmf(
        xrd_data, n_components=n_components
    )

    # Normalize basis vectors
    norm_basis_vector = norm_array(basis_vector, norm=1)

    # Calculate distance matrix
    if distance_method == "DTW":
        dist_matrix = calculate_dtw_matrix(norm_basis_vector, window_size, show_progress)
    elif distance_method == "cosine":
        dist_matrix = calculate_cosine_matrix(norm_basis_vector, show_progress)
    elif distance_method == "correlation":
        dist_matrix = calculate_correlation_matrix(norm_basis_vector, show_progress)
    else:
        raise ValueError(f"Unknown distance method: {distance_method}")

    # Dimension reduction
    embedding = dim_reduction(dist_matrix, method=dim_reduction_method)

    # Clustering
    labels = run_dbscan(embedding, eps=dbscan_eps, min_samples=dbscan_min_samples)

    # Calculate cluster ratios
    norm_coefficient = normalize_coefficient_by_integral(basis_vector, coefficient)
    cluster_ratio = merge_cluster_ratio(norm_coefficient, labels)
    argmax, probabilities = get_argmax_prob(cluster_ratio)

    return {
        'basis_vector': basis_vector,
        'norm_basis_vector': norm_basis_vector,
        'coefficient': coefficient,
        'norm_coefficient': norm_coefficient,
        'reconstruction_error': recon_error,
        'distance_matrix': dist_matrix,
        'embedding': embedding,
        'labels': labels,
        'cluster_ratio': cluster_ratio,
        'argmax': argmax,
        'probabilities': probabilities,
        'two_theta': two_theta,
        'sample_ids': sample_ids,
        'n_clusters': len(set(labels))
    }
