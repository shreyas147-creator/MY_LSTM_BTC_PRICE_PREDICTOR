import numpy as np
import pandas as pd
from scipy.linalg import eigvals


def marchenko_pastur_distribution(N, P, alpha=1.0):
    """
    Calculates the theoretical edge points (min and max) of the Marchenko-Pastur
    distribution for eigenvalues of a sample covariance matrix.

    Args:
        N: Number of samples (rows in the matrix).
        P: Number of features (columns in the matrix).
        alpha: Ratio N/P.

    Returns:
        A tuple (lambda_min, lambda_max) representing the eigenvalue bounds.
    """
    if N == 0 or P == 0:
        raise ValueError("N and P must be positive.")

    alpha_ratio = min(1.0, N / P)

    # The theoretical edge points for the spectrum of W * W / N
    lambda_min = (1.0 - np.sqrt(alpha_ratio)) ** 2
    lambda_max = (1.0 + np.sqrt(alpha_ratio)) ** 2

    print(f"Theoretical MP Bounds (N={N}, P={P}): Min={lambda_min:.4f}, Max={lambda_max:.4f}")
    return lambda_min, lambda_max


def rmt_denoise_covariance(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Performs noise filtering/denoising on a covariance matrix using RMT principles.

    This function estimates the underlying true covariance structure by
    removing eigenvalues that fall within the theoretically predicted noise bulk.

    Args:
        cov_matrix: The observed covariance matrix (must be symmetric positive semi-definite).

    Returns:
        The denoised covariance matrix.
    """
    if not isinstance(cov_matrix, np.ndarray) or cov_matrix.ndim != 2:
        raise TypeError("Input must be a 2D numpy array.")

    # 1. Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # For a real-world scenario, we'd need N and P to run MP.
    # Placeholder assuming some global knowledge of N/P for demonstration.
    N_samples = 100  # Placeholder
    P_features = cov_matrix.shape[0]

    # 2. Determine the theoretical noise band
    try:
        lambda_min, lambda_max = marchenko_pastur_distribution(N_samples, P_features)
    except ValueError as e:
        print(f"Warning: Could not calculate MP bounds: {e}. Proceeding with basic filtering.")
        lambda_min, lambda_max = -1, 1  # Fallback bounds

    # 3. Filter eigenvalues: Keep only those outside the MP bulk (signal)
    signal_mask = (eigenvalues > lambda_max) | (eigenvalues < lambda_min)

    # 4. Reconstruct the denoised matrix
    denoised_eigenvalues = eigenvalues * signal_mask + eigenvalues * (1 - signal_mask) * 0.0
    denoised_matrix = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T

    return denoised_matrix


if __name__ == '__main__':
    print("Testing random_matrix_theory.py module.")

    # Create a mock covariance matrix (e.g., 5x5)
    np.random.seed(42)
    mock_cov = np.random.rand(5, 5)
    mock_cov = mock_cov @ mock_cov.T + np.eye(5) * 0.1  # Ensure positive semi-definite

    print("\n--- Original Mock Covariance Matrix ---")
    print(mock_cov)

    # Perform denoising
    denoised_cov = rmt_denoise_covariance(mock_cov)

    print("\n--- Denoised Covariance Matrix (RMT Filtered) ---")
    print(denoised_cov)