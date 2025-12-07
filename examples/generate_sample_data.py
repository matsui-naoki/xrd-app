"""
Generate sample XRD data for testing and demonstration
"""

import numpy as np
import os


def generate_peak(two_theta: np.ndarray, center: float, intensity: float, width: float) -> np.ndarray:
    """Generate a Gaussian-like XRD peak"""
    return intensity * np.exp(-((two_theta - center) ** 2) / (2 * width ** 2))


def generate_xrd_pattern(two_theta: np.ndarray, pattern_type: str = 'A') -> np.ndarray:
    """
    Generate synthetic XRD pattern

    Pattern types:
    - 'A': Cubic crystal structure (simple peaks)
    - 'B': Tetragonal structure (additional peaks)
    - 'C': Mixed phase (combination of A and B)
    """
    intensity = np.zeros_like(two_theta)

    if pattern_type == 'A':
        # Cubic-like pattern
        peaks = [(20.5, 100, 0.3), (30.2, 80, 0.35), (35.5, 60, 0.4),
                 (42.0, 40, 0.3), (50.5, 30, 0.35), (55.0, 25, 0.4)]
    elif pattern_type == 'B':
        # Tetragonal-like pattern
        peaks = [(22.0, 90, 0.3), (25.5, 70, 0.35), (32.0, 85, 0.3),
                 (38.5, 50, 0.35), (45.0, 45, 0.4), (52.5, 35, 0.35)]
    else:  # 'C' - Mixed
        # Combined pattern
        peaks = [(20.5, 60, 0.3), (22.0, 50, 0.3), (30.2, 45, 0.35),
                 (32.0, 55, 0.3), (38.5, 30, 0.35), (50.5, 20, 0.35)]

    for center, amp, width in peaks:
        intensity += generate_peak(two_theta, center, amp, width)

    return intensity


def add_noise(intensity: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to intensity"""
    noise = np.random.normal(0, noise_level * np.max(intensity), len(intensity))
    return np.maximum(intensity + noise, 0)


def add_background(intensity: np.ndarray, two_theta: np.ndarray,
                   bg_level: float = 10, slope: float = 0.1) -> np.ndarray:
    """Add polynomial background"""
    background = bg_level + slope * (two_theta - two_theta[0])
    background += 5 * np.sin(0.1 * two_theta)  # Some curvature
    return intensity + background


def generate_sample_files(output_dir: str, n_samples: int = 15):
    """
    Generate sample XRD files for testing

    Creates:
    - 5 samples of type A (cluster 1)
    - 5 samples of type B (cluster 2)
    - 5 samples of type C (mixed phase)
    """
    os.makedirs(output_dir, exist_ok=True)

    two_theta = np.linspace(10, 60, 500)

    for i in range(n_samples):
        # Determine pattern type
        if i < 5:
            pattern_type = 'A'
        elif i < 10:
            pattern_type = 'B'
        else:
            pattern_type = 'C'

        # Generate pattern with variations
        intensity = generate_xrd_pattern(two_theta, pattern_type)

        # Add some random variation
        peak_shift = np.random.uniform(-0.2, 0.2)
        shifted_theta = two_theta + peak_shift

        # Add noise and background
        intensity = add_noise(intensity, noise_level=np.random.uniform(0.03, 0.08))
        intensity = add_background(intensity, two_theta,
                                   bg_level=np.random.uniform(5, 15))

        # Scale intensity randomly
        intensity *= np.random.uniform(0.8, 1.2)

        # Save file
        filename = f"sample_{i + 1:03d}.xy"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            f.write(f"# Sample XRD data - Type {pattern_type}\n")
            f.write(f"# 2theta    Intensity\n")
            for x, y in zip(shifted_theta, intensity):
                f.write(f"{x:.4f}\t{y:.4f}\n")

        print(f"Generated: {filename} (Type {pattern_type})")

    print(f"\nGenerated {n_samples} sample files in {output_dir}")


if __name__ == "__main__":
    # Generate sample data in examples directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "sample_data")
    generate_sample_files(output_dir, n_samples=15)
