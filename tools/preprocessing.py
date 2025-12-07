"""
XRD Data Preprocessing Module
Functions for normalizing, trimming, and cleaning XRD data
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

try:
    import pybeads
    HAS_PYBEADS = True
except ImportError:
    HAS_PYBEADS = False


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function for smooth edge processing"""
    return 1 / (1 + np.exp(-x))


def normalize_xrd(xrd_data: Dict[int, List],
                  max_value: float = 100.0) -> Dict[int, List]:
    """
    Normalize XRD intensity to a maximum value

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values
        max_value: Target maximum intensity value (default: 100)

    Returns:
        Normalized XRD data dictionary
    """
    new_dict = {}
    for key, value in xrd_data.items():
        two_theta = list(value[0])
        intensity = np.array(value[1], dtype=float)

        max_intensity = np.max(intensity)
        if max_intensity > 0:
            ratio = max_value / max_intensity
            normalized_intensity = (intensity * ratio).tolist()
        else:
            normalized_intensity = intensity.tolist()

        new_dict[key] = [two_theta, normalized_intensity]

    return new_dict


def trim_2theta(xrd_data: Dict[int, List],
                ranges: List[List[float]] = [[10, 60]]) -> Dict[int, List]:
    """
    Trim XRD data to specified 2theta ranges

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values
        ranges: List of [start, end] ranges to keep

    Returns:
        Trimmed XRD data dictionary
    """
    new_dict = {}

    for key, value in xrd_data.items():
        x_new = []
        y_new = []

        two_theta = value[0]
        intensity = value[1]

        for r in ranges:
            start, end = r[0], r[1]
            for i in range(len(two_theta)):
                xx = two_theta[i]
                if xx > end:
                    break
                elif xx >= start:
                    x_new.append(xx)
                    y_new.append(intensity[i])

        new_dict[key] = [x_new, y_new]

    return new_dict


def remove_background_single(y: np.ndarray,
                             fc: float = 0.004,
                             d: int = 1,
                             r: int = 6,
                             amp: float = 0.8,
                             Nit: int = 15) -> np.ndarray:
    """
    Remove background from a single XRD pattern using BEADS algorithm

    Args:
        y: Intensity array
        fc: High-pass filter cutoff frequency
        d: Filter parameter
        r: Asymmetric penalty parameter
        amp: Amplitude scaling factor
        Nit: Number of iterations

    Returns:
        Background-removed intensity array
    """
    if not HAS_PYBEADS:
        # Fallback: simple polynomial baseline subtraction
        from scipy.signal import savgol_filter
        from scipy.ndimage import minimum_filter1d

        # Estimate baseline using minimum filter + smoothing
        window_size = max(len(y) // 20, 5)
        baseline = minimum_filter1d(y, size=window_size)
        baseline = savgol_filter(baseline, min(window_size * 2 + 1, len(y) // 2 * 2 + 1), 2)
        result = y - baseline
        return np.maximum(result, 0)

    # BEADS algorithm parameters
    lam0 = 0.5 * amp
    lam1 = 5 * amp
    lam2 = 4 * amp
    pen = 'L1_v2'

    # Smooth edge processing
    xscale_l, xscale_r = 100, 100
    dx = 1
    y_difficult_l = y[0] * sigmoid(1 / xscale_l * np.arange(-5 * xscale_l, 5 * xscale_l, dx))
    y_difficult_r = y[-1] * sigmoid(-1 / xscale_r * np.arange(-5 * xscale_r, 5 * xscale_r, dx))
    y_difficult_ext = np.hstack([y_difficult_l, y, y_difficult_r])
    len_l, len_o = len(y_difficult_l), len(y)

    # Run BEADS
    signal_est, bg_est, cost = pybeads.beads(
        y_difficult_ext, d, fc, r, Nit, lam0, lam1, lam2, pen, conv=10
    )

    estimate = signal_est[len_l:len_l + len_o]
    return estimate


def remove_background(xrd_data: Dict[int, List],
                      show_progress: bool = True,
                      **kwargs) -> Dict[int, List]:
    """
    Remove background from all XRD patterns

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values
        show_progress: Whether to show progress bar
        **kwargs: Additional parameters for remove_background_single

    Returns:
        Background-removed XRD data dictionary
    """
    new_dict = {}
    items = list(xrd_data.items())

    iterator = tqdm(items, desc="Removing background") if show_progress else items

    for key, value in iterator:
        two_theta = value[0]
        intensity = np.array(value[1], dtype=float)

        try:
            rm_intensity = remove_background_single(intensity, **kwargs)
            new_dict[key] = [two_theta, rm_intensity.tolist()]
        except Exception as e:
            print(f"Warning: Background removal failed for sample {key}: {e}")
            new_dict[key] = [two_theta, value[1]]

    return new_dict


def remove_xrd_nan(xrd_data: Dict[int, List]) -> Dict[int, List]:
    """
    Remove NaN values from XRD data

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values

    Returns:
        Cleaned XRD data dictionary
    """
    new_dict = {}

    for key, value in xrd_data.items():
        x_new = []
        y_new = []

        for theta, intensity in zip(value[0], value[1]):
            if not np.isnan(theta) and not np.isnan(intensity):
                x_new.append(theta)
                y_new.append(intensity)

        new_dict[key] = [x_new, y_new]

    return new_dict


def negative_to_zero(xrd_data: Dict[int, List]) -> Dict[int, List]:
    """
    Convert negative intensity values to zero

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values

    Returns:
        Processed XRD data dictionary
    """
    new_dict = {}

    for key, value in xrd_data.items():
        two_theta = value[0]
        intensity = [max(0, i) for i in value[1]]
        new_dict[key] = [two_theta, intensity]

    return new_dict


def sort_dict_by_key(xrd_data: Dict[int, List]) -> Dict[int, List]:
    """
    Sort dictionary by keys

    Args:
        xrd_data: Dictionary to sort

    Returns:
        Sorted dictionary
    """
    return {key: xrd_data[key] for key in sorted(xrd_data.keys())}


def delete_by_intensity(xrd_data: Dict[int, List],
                        min_intensity: float = 100) -> Dict[int, List]:
    """
    Remove samples with maximum intensity below threshold

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values
        min_intensity: Minimum maximum intensity to keep

    Returns:
        Filtered XRD data dictionary
    """
    new_dict = {}

    for key, value in xrd_data.items():
        if max(value[1]) > min_intensity:
            new_dict[key] = value

    return new_dict


def interpolate_xrd(xrd_data: Dict[int, List],
                    num_points: int = 1000) -> Dict[int, List]:
    """
    Interpolate XRD data to uniform 2theta spacing

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values
        num_points: Number of interpolation points

    Returns:
        Interpolated XRD data dictionary
    """
    from scipy.interpolate import interp1d

    # Find common 2theta range
    all_min = min(min(v[0]) for v in xrd_data.values())
    all_max = max(max(v[0]) for v in xrd_data.values())

    common_2theta = np.linspace(all_min, all_max, num_points)

    new_dict = {}
    for key, value in xrd_data.items():
        two_theta = np.array(value[0])
        intensity = np.array(value[1])

        # Create interpolation function
        f = interp1d(two_theta, intensity, kind='linear',
                     bounds_error=False, fill_value=0)

        new_intensity = f(common_2theta)
        new_dict[key] = [common_2theta.tolist(), new_intensity.tolist()]

    return new_dict


def smooth_xrd(xrd_data: Dict[int, List],
               window_length: int = 11,
               polyorder: int = 3) -> Dict[int, List]:
    """
    Smooth XRD data using Savitzky-Golay filter

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values
        window_length: Window length for smoothing (must be odd)
        polyorder: Polynomial order

    Returns:
        Smoothed XRD data dictionary
    """
    from scipy.signal import savgol_filter

    if window_length % 2 == 0:
        window_length += 1

    new_dict = {}
    for key, value in xrd_data.items():
        two_theta = value[0]
        intensity = np.array(value[1])

        if len(intensity) > window_length:
            smoothed = savgol_filter(intensity, window_length, polyorder)
            new_dict[key] = [two_theta, smoothed.tolist()]
        else:
            new_dict[key] = [two_theta, value[1]]

    return new_dict


def preprocess_pipeline(xrd_data: Dict[int, List],
                        normalize: bool = True,
                        trim_range: Optional[List[List[float]]] = None,
                        remove_bg: bool = True,
                        remove_nan: bool = True,
                        zero_negative: bool = True,
                        smooth: bool = False,
                        interpolate: bool = False,
                        show_progress: bool = True) -> Dict[int, List]:
    """
    Run full preprocessing pipeline on XRD data

    Args:
        xrd_data: Input XRD data dictionary
        normalize: Whether to normalize intensity
        trim_range: 2theta ranges to keep
        remove_bg: Whether to remove background
        remove_nan: Whether to remove NaN values
        zero_negative: Whether to convert negative values to zero
        smooth: Whether to apply smoothing
        interpolate: Whether to interpolate to uniform spacing
        show_progress: Whether to show progress

    Returns:
        Preprocessed XRD data dictionary
    """
    result = dict(xrd_data)

    if remove_nan:
        result = remove_xrd_nan(result)

    if trim_range:
        result = trim_2theta(result, ranges=trim_range)

    if remove_bg:
        result = remove_background(result, show_progress=show_progress)

    if zero_negative:
        result = negative_to_zero(result)

    if smooth:
        result = smooth_xrd(result)

    if interpolate:
        result = interpolate_xrd(result)

    if normalize:
        result = normalize_xrd(result)

    result = sort_dict_by_key(result)

    return result
