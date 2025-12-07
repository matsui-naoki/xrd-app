"""
XRD Analyzer Helper Functions
"""

import numpy as np
import re
from typing import Optional, Tuple, List, Dict


def format_scientific(value: float, precision: int = 3) -> str:
    """
    Format a number in scientific notation

    Args:
        value: Number to format
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    if value == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)

    if -3 <= exponent <= 3:
        return f"{value:.{precision}f}"
    else:
        return f"{mantissa:.{precision}f}×10^{exponent}"


def extract_number_from_filename(filename: str) -> Optional[int]:
    """
    Extract a number from a filename

    Args:
        filename: Filename string

    Returns:
        Extracted number or None
    """
    # Remove extension
    name = filename.rsplit('.', 1)[0]

    # Try to find number patterns
    patterns = [
        r'_(\d+)$',           # number at end after underscore
        r'-(\d+)$',           # number at end after hyphen
        r'(\d+)$',            # number at end
        r'_(\d+)_',           # number between underscores
        r'sample_?(\d+)',     # sample + number
        r'#(\d+)',            # hash + number
    ]

    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def generate_sample_info(filenames: List[str]) -> Dict[str, Dict]:
    """
    Generate sample information from filenames

    Args:
        filenames: List of filenames

    Returns:
        Dictionary with sample info
    """
    info = {}
    for filename in filenames:
        sample_id = extract_number_from_filename(filename)
        info[filename] = {
            'sample_id': sample_id if sample_id else len(info) + 1,
            'original_name': filename
        }
    return info


def ternary_coords(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    Convert ternary coordinates (a, b, c) to Cartesian (x, y)

    Args:
        a, b, c: Ternary coordinates (should sum to 1)

    Returns:
        Tuple of (x, y) Cartesian coordinates
    """
    x = 1 - a / 2.0 - b
    y = a * np.sqrt(3.0) / 2.0
    return x, y


def reverse_ternary_coords(x: float, y: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian (x, y) to ternary coordinates (a, b, c)

    Args:
        x, y: Cartesian coordinates

    Returns:
        Tuple of (a, b, c) ternary coordinates
    """
    a = 2 * y / np.sqrt(3)
    b = 1 - x - a / 2
    c = 1 - a - b
    return a, b, c


def ternary_abc2coords(data: np.ndarray) -> np.ndarray:
    """
    Convert array of ternary coordinates to Cartesian

    Args:
        data: Array of shape (n, 3) with ternary coordinates

    Returns:
        Array of shape (n, 2) with Cartesian coordinates
    """
    data = np.array(data)

    # Normalize to sum = 1
    row_sums = data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = data / row_sums

    # Convert to Cartesian
    coords = np.apply_along_axis(lambda row: ternary_coords(*row), 1, normalized)
    return coords


def make_ternary_grid(resolution: int = 101) -> List[List[float]]:
    """
    Create a grid of ternary coordinates

    Args:
        resolution: Number of points per axis

    Returns:
        List of [a, b, c] coordinates
    """
    coords = []
    for x in range(resolution):
        for y in range(resolution):
            z = resolution - 1 - x - y
            if z >= 0:
                a = x / (resolution - 1)
                b = y / (resolution - 1)
                c = z / (resolution - 1)
                coords.append([a, b, c])
    return coords


def calculate_peak_positions(two_theta: np.ndarray,
                            intensity: np.ndarray,
                            min_height: float = 0.1,
                            min_distance: int = 5) -> List[float]:
    """
    Find peak positions in XRD pattern

    Args:
        two_theta: 2theta values
        intensity: Intensity values
        min_height: Minimum peak height (relative to max)
        min_distance: Minimum distance between peaks (in data points)

    Returns:
        List of peak 2theta positions
    """
    from scipy.signal import find_peaks

    # Normalize intensity
    norm_intensity = intensity / np.max(intensity)

    # Find peaks
    peaks, _ = find_peaks(norm_intensity, height=min_height, distance=min_distance)

    return two_theta[peaks].tolist()


def calculate_crystallite_size(two_theta: float,
                               fwhm: float,
                               wavelength: float = 1.5406,
                               shape_factor: float = 0.9) -> float:
    """
    Calculate crystallite size using Scherrer equation

    Args:
        two_theta: Peak position in degrees
        fwhm: Full width at half maximum in degrees
        wavelength: X-ray wavelength in Angstroms (default: Cu Kα)
        shape_factor: Scherrer constant (default: 0.9)

    Returns:
        Crystallite size in nm
    """
    # Convert to radians
    theta_rad = np.radians(two_theta / 2)
    fwhm_rad = np.radians(fwhm)

    # Scherrer equation: D = K * λ / (β * cos(θ))
    size = (shape_factor * wavelength) / (fwhm_rad * np.cos(theta_rad))

    return size * 0.1  # Convert Angstroms to nm


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Apply softmax function

    Args:
        x: Input array

    Returns:
        Softmax output
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def save_results_to_csv(results: Dict, filepath: str) -> None:
    """
    Save analysis results to CSV

    Args:
        results: Analysis results dictionary
        filepath: Output file path
    """
    import pandas as pd

    data = {
        'Sample ID': results['sample_ids'],
        'Argmax Cluster': [x + 1 for x in results['argmax']],
    }

    # Add probability columns
    n_clusters = len(results['probabilities'][0])
    for i in range(n_clusters):
        data[f'Cluster {i+1} Prob'] = [p[i] for p in results['probabilities']]

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def load_results_from_csv(filepath: str) -> Dict:
    """
    Load analysis results from CSV

    Args:
        filepath: Input file path

    Returns:
        Results dictionary
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    results = {
        'sample_ids': df['Sample ID'].tolist(),
        'argmax': [x - 1 for x in df['Argmax Cluster'].tolist()],
    }

    # Extract probability columns
    prob_cols = [col for col in df.columns if 'Prob' in col]
    probabilities = df[prob_cols].values.tolist()
    results['probabilities'] = probabilities

    return results
