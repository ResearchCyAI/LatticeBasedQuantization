"""
Utility Functions Module

This module provides utility functions and lattice generator matrices for the
LatticeQuant library. It contains common functions used across all modules
for lattice operations, analysis, and computation.

Lattice Generators:
    - get_d4(), get_a2(), get_e8(): High-dimensional lattice matrices
    - get_z2(), get_z3(): Integer lattice matrices
    - get_d2(), get_d3(): D-series lattice matrices

Closest Point Algorithms:
    - closest_point_Dn(): D_n lattice closest point algorithm
    - closest_point_E8(): E_8 lattice closest point algorithm
    - closest_point_A2(): A_2 lattice closest point algorithm
    - custom_round(): Custom rounding function for lattice quantization

Analysis Functions:
    - calculate_mse(): Mean squared error calculation
    - calculate_t_entropy(): Entropy calculation for overload handling
    - precompute_hq_lut(): Lookup table generation
    - calculate_weighted_sum(): Weighted inner product estimation

Constants:
    - SIG_D3, SIG_D4, SIG_E8: Lattice-specific parameters

This module serves as a central repository for common functionality used
by the quantizers, applications, and adaptive modules.
"""

import numpy as np

# Constants for lattice parameters
SIG_D3 = 3 / 24
SIG_D4 = np.sqrt(2) * 0.076602
SIG_E8 = (1 / 8) * (929 / 1620)


def calculate_mse(x, x_hat):
    """
    Calculate the Mean Squared Error (MSE) between two vectors.

    Parameters:
    -----------
    x : numpy.ndarray
        Original vector.
    x_hat : numpy.ndarray
        Reconstructed/estimated vector.

    Returns:
    --------
    float
        The mean squared error between x and x_hat.
    """
    return np.mean((x - x_hat) ** 2)


def get_z2():
    """
    Get the generator matrix for the Z^2 lattice.

    The Z^2 lattice is the standard integer lattice in 2D.

    Returns:
    --------
    numpy.ndarray
        2x2 identity matrix representing the Z^2 lattice.
    """
    return np.array([[1, 0], [0, 1]])


def get_z3():
    """
    Get the generator matrix for the Z^3 lattice.

    The Z^3 lattice is the standard integer lattice in 3D.

    Returns:
    --------
    numpy.ndarray
        3x3 identity matrix representing the Z^3 lattice.
    """
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def get_a2():
    """
    Get the generator matrix for the A_2 lattice.

    The A_2 lattice is a 2D hexagonal lattice with excellent
    packing properties.

    Returns:
    --------
    numpy.ndarray
        2x2 generator matrix for the A_2 lattice.

    Notes:
    ------
    The A_2 lattice has a hexagonal structure and is optimal
    for 2D sphere packing.
    """
    return np.array([[1, 0], [0.5, np.sqrt(3) / 2]]).T


def get_d2():
    """
    Get the generator matrix for the D_2 lattice.

    The D_2 lattice is a 2D lattice where the sum of coordinates
    must be even.

    Returns:
    --------
    numpy.ndarray
        2x2 generator matrix for the D_2 lattice.
    """
    return np.array([[1, -1], [2, 0]]).T


def get_d3():
    """
    Get the generator matrix for the D_3 lattice.

    The D_3 lattice is a 3D lattice where the sum of coordinates
    must be even.

    Returns:
    --------
    numpy.ndarray
        3x3 generator matrix for the D_3 lattice.
    """
    return np.array([[1, -1, 0], [0, 1, -1], [0, 1, 1]]).T


def get_d4():
    """
    Get the generator matrix for the D_4 lattice.

    The D_4 lattice is a 4D lattice where the sum of coordinates
    must be even. It has excellent packing properties in 4D.

    Returns:
    --------
    numpy.ndarray
        4x4 generator matrix for the D_4 lattice.

    Notes:
    ------
    The D_4 lattice is optimal for 4D sphere packing and is
    commonly used in lattice quantization applications.
    """
    return np.array([[-1, -1, 0, 0], [1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]]).T


def get_e8():
    """
    Get the generator matrix for the E_8 lattice.

    The E_8 lattice is an 8D lattice with exceptional properties.
    It is optimal for 8D sphere packing and has many applications
    in coding theory and quantization.

    Returns:
    --------
    numpy.ndarray
        8x8 generator matrix for the E_8 lattice.

    Notes:
    ------
    The E_8 lattice is constructed from the D_8 lattice and a coset.
    It has the highest known packing density in 8D.
    """
    return np.array(
        [
            [2, 0, 0, 0, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 1, 0, 0],
            [0, 0, 0, 0, 0, -1, 1, 0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ]
    ).T


def calculate_t_entropy(T_values, q):
    """
    Calculate the entropy of the T values (overload counts) for rate calculation.

    Parameters:
    -----------
    T_values : list or numpy.ndarray
        List of T values (overload counts) from quantization.
    q : int
        The quantization parameter.

    Returns:
    --------
    tuple
        (H_T, T_counts) where H_T is the entropy in bits and T_counts
        is the count of each T value.
    """
    T_counts = np.bincount(T_values, minlength=q**2)
    T_probs = T_counts / np.sum(T_counts)
    H_T = -sum(p * np.log2(p) for p in T_probs if p > 0)
    return H_T, T_counts


def precompute_hq_lut(G, Q_nn, q, m, eps):
    """
    Precompute a lookup table for hierarchical quantization.

    This function creates a lookup table that maps pairs of encoding vectors
    to their inner products, which can be used for efficient inner product
    estimation in hierarchical quantization.

    Parameters:
    -----------
    G : numpy.ndarray
        Generator matrix for the lattice.
    Q_nn : function
        Closest point function for the lattice.
    q : int
        Quantization parameter.
    m : int
        Number of hierarchical levels.
    eps : float
        Small perturbation parameter.

    Returns:
    --------
    dict
        Lookup table mapping (enc1, enc2) tuples to inner product values.

    Notes:
    ------
    The lookup table is used to accelerate inner product calculations
    in hierarchical quantization by precomputing all possible inner
    products between quantized vectors.
    """
    from .lattice.hnlq import HNLQ as HQuantizer

    hq = HQuantizer(
        G=G, Q_nn=Q_nn, q=q, beta=1, alpha=1, M=m, eps=eps, dither=np.zeros(len(G))
    )
    codebook = hq.create_q_codebook(with_dither=False)
    lookup_table = {}
    for enc1, lattice_point1 in codebook.items():
        for enc2, lattice_point2 in codebook.items():
            inner_product = np.dot(lattice_point1, lattice_point2)
            lookup_table[(enc1, enc2)] = inner_product
    return lookup_table


def calculate_weighted_sum(a_list, b_list, lut, q):
    """
    Calculate the weighted sum for given encoding vectors using a lookup table.

    This function computes the weighted sum of inner products between
    pairs of encoding vectors, which is used in hierarchical quantization
    for inner product estimation.

    Parameters:
    -----------
    a_list : list
        List of encoding vectors for the first vector.
    b_list : list
        List of encoding vectors for the second vector.
    lut : dict
        Lookup table mapping (enc1, enc2) to inner product values.
    q : int
        Quantization parameter.

    Returns:
    --------
    float
        The weighted sum of inner products.

    Raises:
    -------
    ValueError
        If a_list and b_list have different lengths.

    Notes:
    ------
    The weighted sum is computed as:
    sum_{i,j} q^(i+j) * lut[(a_list[i], b_list[j])]
    where the weights q^(i+j) correspond to the hierarchical levels.
    """
    k = len(a_list)
    if len(b_list) != k:
        raise ValueError("a_list and b_list must have the same length.")

    total_sum = 0
    for i in range(k):
        for j in range(k):
            weight = q ** (i + j)
            total_sum += weight * lut[(tuple(a_list[i]), tuple(b_list[j]))]

    return total_sum


# lattice agnostic dither (for breaking ties. not to be confused with the subtractive dither)


def generate_tie_dither(d, beta=1.0, Rin=0.5, magnitude="auto"):
    # a constant, sample independent dither, to break ties

    # direction with irrational components -> avoids alignment with faces
    irr = np.array([np.sqrt(p) for p in [2, 3, 5, 7, 11, 13, 17, 19][:d]])
    u = (irr - np.floor(irr)) - 0.5
    u /= np.linalg.norm(u)

    if magnitude == "auto":
        # very small relative to scale & lattice packing radius
        eta = 2.0**-40  # use 2**-20 if float32 end-to-end
        delta = eta * beta * Rin
    else:
        delta = float(magnitude)

    return delta * u  # add to x before Q_L(x)


# usage:
# eps = fixed_tie_dither(d=4, beta=beta, Rin=1/np.sqrt(2))  # e.g., for D4
# z = Q_L(x + eps)  # your lattice nearest-point algorithm


# Closest Point Algorithms


def custom_round_old(x):
    """
    DEPRECATED: Old custom rounding function that handles edge cases for lattice quantization.

    This function implements the original rounding scheme that ensures consistent behavior
    at boundary points (0.5) for lattice quantization algorithms.

    Parameters:
    -----------
    x : float or numpy.ndarray
        The value(s) to be rounded. Can be a scalar or array.

    Returns:
    --------
    float or numpy.ndarray
        The rounded value(s) using the custom rounding scheme.

    Notes:
    ------
    DEPRECATED: Use custom_round() instead.
    For positive values: rounds down if fractional part is exactly 0.5,
    otherwise rounds to nearest integer.
    For negative values: rounds up if fractional part is exactly 0.5,
    otherwise rounds to nearest integer.
    """
    if isinstance(x, np.ndarray):
        return np.array([custom_round_old(val) for val in x])
    else:
        if x > 0:
            return (
                np.floor(x + 0.5) - 1 if (x - np.floor(x)) == 0.5 else np.floor(x + 0.5)
            )
        else:
            return np.ceil(x - 0.5) + 1 if (np.ceil(x) - x) == 0.5 else np.ceil(x - 0.5)


def custom_round(x, tiny=None):
    """
    Custom rounding function that handles edge cases for lattice quantization.

    This function implements a rounding scheme that ensures consistent behavior
    at boundary points (0.5) for lattice quantization algorithms by nudging
    toward zero so exact .5 falls to the nearer-integer toward zero.

    Parameters:
    -----------
    x : float or numpy.ndarray
        The value(s) to be rounded. Can be a scalar or array.
    tiny : float, optional
        A microscopic nudge relative to dtype. If None, uses machine epsilon.

    Returns:
    --------
    float or numpy.ndarray
        The rounded value(s) using the custom rounding scheme.

    Notes:
    ------
    Uses round_half_to_zero approach: nudges toward zero so exact .5 falls
    to the nearer-integer toward zero, then rounds to nearest via floor(x+0.5).
    """
    x = np.asarray(x)

    if tiny is None:
        # choose a microscopic nudge relative to dtype
        tiny = np.finfo(x.dtype if x.dtype.kind in "fc" else np.float64).eps

    # nudge toward zero so exact .5 falls to the nearer-integer toward zero
    y = x - np.sign(x) * tiny

    # round-to-nearest via floor(x+0.5) works for all signs after the nudge
    result = np.floor(y + 0.5)

    # Preserve original type if scalar, otherwise return array
    if x.ndim == 0:  # scalar
        return result.item()
    else:
        return result


def g_x(x):
    """
    Compute g(x) by rounding the vector x to the nearest integers,
    but flip the rounding for the coordinate farthest from an integer.

    This is a helper function for the D_n lattice closest point algorithm.

    Parameters:
    -----------
    x : numpy.ndarray
        Input vector to be processed.

    Returns:
    --------
    numpy.ndarray
        Modified rounded vector with one coordinate flipped to ensure
        the sum of components has the desired parity.

    Notes:
    ------
    This function is used in the D_n lattice algorithm to find an alternative
    rounding when the standard rounding doesn't satisfy the D_n lattice constraints.
    """
    f_x = custom_round(x)
    delta = np.abs(x - f_x)
    k = np.argmax(delta)
    g_x_ = f_x.copy()

    # Ensure we're working with scalar values
    # x_k = x.flat[k] if hasattr(x, 'flat') else x[k]
    # f_x_k = f_x.flat[k] if hasattr(f_x, 'flat') else f_x[k]

    x_k = x[k]
    f_x_k = f_x[k]

    if x_k >= 0:
        g_x_[k] = f_x_k + 1 if f_x_k < x_k else f_x_k - 1
    else:
        g_x_[k] = f_x_k + 1 if f_x_k <= x_k else f_x_k - 1

    return g_x_


def closest_point_Dn(x):
    """
    Find the closest point in the D_n lattice for a given vector x.

    The D_n lattice consists of all integer points where the sum of coordinates
    is even. This algorithm finds the closest such point to the input vector x.

    Parameters:
    -----------
    x : numpy.ndarray
        Input vector of dimension n.

    Returns:
    --------
    numpy.ndarray
        The closest point in the D_n lattice to x.

    Notes:
    ------
    The algorithm works by:
    1. Rounding x to the nearest integer vector
    2. If the sum is even, this is the closest point
    3. If the sum is odd, flip the rounding for the coordinate farthest
       from an integer to get a valid D_n lattice point

    References:
    -----------
    Conway, J. H., & Sloane, N. J. A. (1982). Fast quantizing and decoding
    algorithms for lattice quantizers and codes. IEEE Transactions on
    Information Theory, 28(2), 227-232.
    """
    f_x = custom_round(x)
    g_x_res = g_x(x)
    return f_x if np.sum(f_x) % 2 == 0 else g_x_res


def closest_point_E8(x):
    """
    Find the closest point in the E_8 lattice for a given vector x.

    The E_8 lattice is an 8-dimensional lattice that can be constructed
    from the D_8 lattice and a coset. This algorithm finds the closest
    E_8 lattice point to the input vector.

    Parameters:
    -----------
    x : numpy.ndarray
        Input vector of dimension 8.

    Returns:
    --------
    numpy.ndarray
        The closest point in the E_8 lattice to x.

    Notes:
    ------
    The E_8 lattice is constructed as the union of D_8 and D_8 + (0.5)^8.
    The algorithm:
    1. Finds the closest point in D_8 to x
    2. Finds the closest point in D_8 + (0.5)^8 to x
    3. Returns the closer of the two points

    References:
    -----------
    Conway, J. H., & Sloane, N. J. A. (1982). Fast quantizing and decoding
    algorithms for lattice quantizers and codes. IEEE Transactions on
    Information Theory, 28(2), 227-232.
    """
    f_x = custom_round(x)
    y_0 = f_x if np.sum(f_x) % 2 == 0 else g_x(x)

    f_x_shifted = custom_round(x - 0.5)
    g_x_shifted = g_x(x - 0.5)

    y_1 = f_x_shifted + 0.5 if np.sum(f_x_shifted) % 2 == 0 else g_x_shifted + 0.5

    if np.linalg.norm(x - y_0) < np.linalg.norm(x - y_1):
        return y_0
    else:
        return y_1


def upscale(u):
    """
    Upscale a 2D vector to 3D space for A_2 lattice processing.

    This function transforms a 2D vector into 3D space using a specific
    transformation matrix that preserves the A_2 lattice structure.

    Parameters:
    -----------
    u : numpy.ndarray
        Input 2D vector.

    Returns:
    --------
    numpy.ndarray
        3D vector obtained by applying the upscaling transformation.

    Notes:
    ------
    The transformation matrix M is:
    [[1, 0, -1],
     [1/√3, -2/√3, 1/√3]]
    """
    M = np.array([[1, 0, -1], [1 / np.sqrt(3), -2 / np.sqrt(3), 1 / np.sqrt(3)]])
    return np.dot(u, M)


def downscale(x):
    """
    Downscale a 3D vector back to 2D space for A_2 lattice processing.

    This function transforms a 3D vector back to 2D space using the
    transpose of the upscaling transformation matrix.

    Parameters:
    -----------
    x : numpy.ndarray
        Input 3D vector.

    Returns:
    --------
    numpy.ndarray
        2D vector obtained by applying the downscaling transformation.

    Notes:
    ------
    This is the inverse operation of upscale(), using the transpose
    of the transformation matrix multiplied by 0.5.
    """
    M_t = np.array([[1, 0, -1], [1 / np.sqrt(3), -2 / np.sqrt(3), 1 / np.sqrt(3)]]).T
    return 0.5 * np.dot(x, M_t)


def closest_point_A2(u):
    """
    Find the closest point in the A_2 lattice for a given vector u.

    The A_2 lattice is a 2D hexagonal lattice. This algorithm finds
    the closest A_2 lattice point to the input vector using a 3D
    embedding approach.

    Parameters:
    -----------
    u : numpy.ndarray
        Input 2D vector.

    Returns:
    --------
    numpy.ndarray
        The closest point in the A_2 lattice to u.

    Notes:
    ------
    The algorithm works by:
    1. Upscaling the 2D vector to 3D space
    2. Finding the closest point in the 3D lattice with sum constraint
    3. Downscaling the result back to 2D space

    References:
    -----------
    Conway, J. H., & Sloane, N. J. A. (1982). Fast quantizing and decoding
    algorithms for lattice quantizers and codes. IEEE Transactions on
    Information Theory, 28(2), 227-232.
    """
    x = upscale(u)
    s = np.sum(x)
    x_p = x - (s / len(x)) * np.array([1, 1, 1])
    f_x_p = custom_round(x_p)
    delta = int(np.sum(f_x_p))

    distances = x - f_x_p
    sorted_indices = np.argsort(distances)

    if delta == 0:
        return downscale(f_x_p)
    elif delta > 0:
        for i in range(delta):
            f_x_p[sorted_indices[i]] -= 1
    elif delta < 0:
        for i in range(-delta):
            f_x_p[sorted_indices[-i - 1]] += 1

    return downscale(f_x_p)
