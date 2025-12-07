"""
Help Texts for XRD Analyzer
"""

HELP_TEXTS = {
    # File upload
    'file_upload': """
    **Supported file formats:**
    - `.xy`, `.txt`: Two-column format (2θ, intensity)
    - `.csv`: CSV with 2θ and intensity columns
    - `.ras`: Rigaku RAS format

    You can upload multiple files at once for batch processing.
    """,

    # Preprocessing
    'normalize': """
    **Normalization:**
    Scales all intensities so that the maximum intensity equals 100.
    This allows comparison between different samples with varying absolute intensities.
    """,

    'trim_2theta': """
    **2θ Range Trimming:**
    Remove data points outside the specified 2θ range.
    Useful for removing low-angle noise or high-angle regions with poor signal.

    You can specify multiple ranges to keep, separated by commas.
    Example: `10-30, 35-60` keeps data in ranges 10-30° and 35-60°.
    """,

    'background_removal': """
    **Background Removal:**
    Removes the baseline/background from XRD patterns using the BEADS algorithm.

    **Parameters:**
    - **fc (cutoff frequency):** Controls the smoothness of baseline estimation.
      Smaller values = smoother baseline (default: 0.004)
    - **r (asymmetry):** Controls asymmetric penalty for baseline fitting (default: 6)
    """,

    'smoothing': """
    **Smoothing:**
    Apply Savitzky-Golay filter to smooth the XRD pattern.

    **Parameters:**
    - **Window length:** Number of points in smoothing window (must be odd)
    - **Polynomial order:** Order of polynomial used for fitting
    """,

    # NMF
    'nmf': """
    **Non-negative Matrix Factorization (NMF):**
    Decomposes XRD patterns into a set of basis patterns (components) and their coefficients.

    Each sample's pattern = Σ (coefficient_i × basis_pattern_i)

    **Parameters:**
    - **Number of components:** How many basis patterns to extract.
      Use the error plot to find optimal value (elbow point).
    """,

    'nmf_components': """
    **Number of Components:**
    The number of basis patterns (components) to extract.

    - Too few: Cannot capture all pattern variations
    - Too many: Overfitting, noisy components

    Use the reconstruction error plot to find the optimal number.
    Look for the "elbow point" where error reduction slows down.
    """,

    'reconstruction_error': """
    **Reconstruction Error:**
    Measures how well the NMF approximation matches the original data.

    Error (%) = ||Original - Reconstructed|| / ||Original|| × 100

    Lower is better, but very low values may indicate overfitting.
    """,

    # Clustering
    'distance_method': """
    **Distance Calculation Methods:**

    - **DTW (Dynamic Time Warping):** Handles peak shifts well.
      Best for comparing patterns with slight 2θ variations.
    - **Cosine:** Fast, measures angular similarity.
      Good for comparing relative peak intensities.
    - **Correlation:** Uses Pearson + Spearman correlation.
      Robust to scaling differences.
    """,

    'dtw_window': """
    **DTW Window Size:**
    Maximum allowed warping (shift) between patterns.

    - Larger values: More flexibility in matching shifted peaks
    - Smaller values: Stricter matching, faster computation

    Typical values: 10-50
    """,

    'dim_reduction': """
    **Dimension Reduction:**
    Reduces high-dimensional distance matrix to 2D for visualization.

    - **MDS:** Preserves global distances well. Recommended for most cases.
    - **t-SNE:** Good for local structure, may distort global distances.
    - **UMAP:** Fast, preserves both local and global structure.
    """,

    'dbscan': """
    **DBSCAN Clustering:**
    Groups similar basis patterns into clusters.

    **Parameters:**
    - **eps:** Maximum distance between points in a cluster.
      Smaller = more clusters
    - **min_samples:** Minimum points to form a cluster.
      Usually 1-3 for small datasets
    """,

    'dbscan_eps': """
    **DBSCAN eps Parameter:**
    The maximum distance between two samples to be considered in the same neighborhood.

    - Smaller values: More, smaller clusters
    - Larger values: Fewer, larger clusters

    Look at the cluster scatter plot to adjust.
    """,

    # Mapping
    'ternary_plot': """
    **Ternary Phase Diagram:**
    Visualizes composition-structure relationships for 3-component systems.

    Each point represents a sample, colored by its assigned cluster.
    This reveals phase regions in the composition space.
    """,

    'cluster_ratio': """
    **Cluster Ratio:**
    For each sample, shows the relative contribution of each cluster.

    Calculated by summing NMF coefficients for components belonging to each cluster.
    """,

    'probability_map': """
    **Probability Distribution:**
    Shows the probability of each sample belonging to each cluster.

    Derived from normalized cluster ratios.
    Higher values indicate stronger association with that cluster.
    """,

    # General
    'session_save': """
    **Session Save/Load:**
    Save your current analysis state including:
    - Loaded files
    - Preprocessing settings
    - Analysis results

    Sessions are saved as JSON files that can be loaded later.
    """,

    'export': """
    **Export Options:**
    - **CSV:** Results table with sample IDs, clusters, and probabilities
    - **HTML:** Interactive plots for viewing in browser
    - **PNG:** Static images of plots
    """,
}


def get_help(topic: str) -> str:
    """
    Get help text for a topic

    Args:
        topic: Topic key

    Returns:
        Help text string
    """
    return HELP_TEXTS.get(topic, "No help available for this topic.")
