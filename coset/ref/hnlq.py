import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .nlq import NLQ as NQ
from .utils import custom_round, generate_tie_dither


@dataclass
class HNLQConfig:
    """
    Configuration class for hierarchical nested lattice quantizer parameters.

    This class provides a structured way to manage HNLQ parameters
    with built-in validation and default values.

    Attributes:
    -----------
    lattice_type : str
        Type of lattice ('D4', 'E8', 'A2', 'Z2', 'Z3').
    q : int
        Quantization parameter (alphabet size).
    beta : float
        Scaling parameter for quantization.
    alpha : float
        Scaling parameter for overload handling.
    eps : float
        Small perturbation parameter.
    M : int
        Number of hierarchical levels.
    overload : bool
        Whether to handle overload by scaling.
    decoding : str
        Default decoding method ('full', 'coarse_to_fine', 'progressive').
    max_scaling_iterations : int
        Maximum number of scaling iterations to prevent infinite loops.
    with_tie_dither : bool
        Whether to add dither to the input for tie breaking.
    with_dither : bool
        Whether to add dither to the input for randomized quantization.
    """

    lattice_type: str
    q: int
    M: int
    beta: float = 1.0
    alpha: float = 1.0
    eps: float = 1e-8
    overload: bool = True
    decoding: str = "full"
    max_scaling_iterations: int = 10
    with_tie_dither: bool = True
    with_dither: bool = False

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.lattice_type not in ["D4", "E8", "A2", "Z2", "Z3"]:
            raise ValueError(f"Unsupported lattice type: {self.lattice_type}")
        if self.q <= 0:
            raise ValueError("Quantization parameter q must be positive")
        if self.beta <= 0:
            raise ValueError("Scaling parameter beta must be positive")
        if self.alpha <= 0:
            raise ValueError("Scaling parameter alpha must be positive")
        if self.M <= 0:
            raise ValueError("Number of hierarchical levels M must be positive")
        if self.max_scaling_iterations <= 0:
            raise ValueError("max_scaling_iterations must be positive")
        if self.decoding not in ["full", "coarse_to_fine", "progressive"]:
            raise ValueError(f"Unknown decoding method: {self.decoding}")
        if not isinstance(self.with_tie_dither, bool):
            raise ValueError("with_tie_dither must be a Boolean")
        if not isinstance(self.with_dither, bool):
            raise ValueError("with_dither must be a Boolean")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HNLQConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "lattice_type": self.lattice_type,
            "q": self.q,
            "beta": self.beta,
            "alpha": self.alpha,
            "eps": self.eps,
            "M": self.M,
            "overload": self.overload,
            "decoding": self.decoding,
            "max_scaling_iterations": self.max_scaling_iterations,
            "with_tie_dither": self.with_tie_dither,
            "with_dither": self.with_dither,
        }


class HNLQ:
    """
    Hierarchical Nested Lattice Quantizer implementing multi-level quantization.

    This class implements a hierarchical quantization approach that uses multiple
    levels of quantization to achieve better rate-distortion performance. The
    hierarchical structure allows for successive refinement and efficient inner
    product estimation using small lookup tables.

    Note:
    ----
    To use single-level quantization (equivalent to NLQ), set M=1 in the configuration.
    The NLQ class is deprecated and will not be maintained.

    Attributes:
    -----------
    G : numpy.ndarray
        Generator matrix for the lattice.
    Q_nn : Callable
        Closest point function for the lattice.
    config : HNLQConfig
        Configuration object containing all quantizer parameters.
    dither : numpy.ndarray
        Dither vector for randomized quantization.
    G_inv : numpy.ndarray
        Inverse of the generator matrix.

    Notes:
    ------
    The hierarchical quantizer uses M levels of quantization, where each level
    provides a refinement of the previous level. This approach enables efficient
    inner product estimation and achieves better rate-distortion performance
    compared to single-level quantization.
    """

    def __init__(
        self,
        config: Union[HNLQConfig, Dict[str, Any]],
        G: Optional[np.ndarray] = None,
        Q_nn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs,
    ):
        """
        Initialize the Hierarchical Nested Lattice Quantizer.

        Parameters:
        -----------
        config : HNLQConfig or Dict[str, Any]
            Configuration object or dictionary with quantizer parameters.
        G : numpy.ndarray, optional
            Generator matrix for the lattice. If not provided, will be loaded based on lattice_type.
        Q_nn : Callable, optional
            Closest point function for the lattice. If not provided, will be loaded based on lattice_type.
        **kwargs : Additional parameters (for backward compatibility)
        """
        # Handle configuration
        if isinstance(config, dict):
            config = HNLQConfig.from_dict(config)
        elif not isinstance(config, HNLQConfig):
            raise ValueError("config must be HNLQConfig or dict")

        self.config = config

        # Load lattice components if not provided
        if G is None or Q_nn is None:
            G, Q_nn = self._load_lattice_components(config.lattice_type)

        if G.shape[0] != G.shape[1]:
            raise ValueError("Generator matrix G must be square")

        self.G = G

        # Handle tie dither for breaking ties in lattice quantization
        if self.config.with_tie_dither:
            self._original_eps = generate_tie_dither(G.shape[0])
        else:
            self._original_eps = config.eps

        # Set up Q_nn with appropriate tie dither
        self.Q_nn = lambda x: Q_nn(x + self._original_eps)

        # Initialize dither as zeros - will be generated on demand when with_dither=True
        self.dither = np.zeros(self.G.shape[0])

        self.G_inv = np.linalg.inv(G)

        # Precompute some values for efficiency
        self._dim = G.shape[0]

    def _load_lattice_components(
        self, lattice_type: str
    ) -> Tuple[np.ndarray, Callable]:
        """
        Load generator matrix and closest point function for the specified lattice type.

        Parameters:
        -----------
        lattice_type : str
            Type of lattice ('D4', 'E8', 'A2', 'Z2', 'Z3').

        Returns:
        --------
        tuple
            (G, Q_nn) where G is the generator matrix and Q_nn is the closest point function.
        """
        from .utils import (
            closest_point_Dn,
            closest_point_E8,
            get_a2,
            get_d4,
            get_e8,
            get_z2,
            get_z3,
        )

        if lattice_type == "D4":
            return get_d4(), closest_point_Dn
        elif lattice_type == "E8":
            return get_e8(), closest_point_E8
        elif lattice_type == "A2":
            return get_a2(), closest_point_Dn
        elif lattice_type == "Z2":

            def closest_point_zn(x):
                return np.floor(x + 0.5)

            return get_z2(), closest_point_zn
        elif lattice_type == "Z3":

            def closest_point_zn(x):
                return np.floor(x + 0.5)

            return get_z3(), closest_point_zn
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")

    # ============================================================================
    # Properties
    # ============================================================================

    @property
    def lattice_type(self) -> str:
        """Get lattice type."""
        return self.config.lattice_type

    @property
    def q(self) -> int:
        """Get quantization parameter."""
        return self.config.q

    @property
    def beta(self) -> float:
        """Get scaling parameter."""
        return self.config.beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Set scaling parameter."""
        self.config.beta = value

    @property
    def alpha(self) -> float:
        """Get overload scaling parameter."""
        return self.config.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set overload scaling parameter."""
        self.config.alpha = value

    @property
    def eps(self) -> float:
        """Get perturbation parameter."""
        return self.config.eps

    @property
    def M(self) -> int:
        """Get number of hierarchical levels."""
        return self.config.M

    @property
    def overload(self) -> bool:
        """Get overload handling flag."""
        return self.config.overload

    @property
    def decoding(self) -> str:
        """Get default decoding method."""
        return self.config.decoding

    @property
    def with_tie_dither(self) -> bool:
        """Get tie dither flag."""
        return self.config.with_tie_dither

    @property
    def with_dither(self) -> bool:
        """Get dither flag."""
        return self.config.with_dither

    # ============================================================================
    # Encoding Methods
    # ============================================================================

    def _encode(
        self, x: np.ndarray, with_dither: bool
    ) -> Tuple[Tuple[np.ndarray, ...], bool]:
        """
        Internal encoding function that performs hierarchical quantization.

        This method implements the core hierarchical encoding algorithm,
        producing M levels of encoding vectors.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to be quantized.
        with_dither : bool
            Whether to apply dithering during quantization.

        Returns:
        --------
        Tuple[Tuple[numpy.ndarray, ...], bool]
            (encoding_vectors, overload_error) where encoding_vectors is a
            tuple of M encoding vectors and overload_error indicates if
            overload occurred.
        """
        # Ensure x is a column vector
        x = np.asarray(x).flatten()
        x = x / self.beta
        if with_dither:
            x = x + self.dither.flatten()
        x_l = x
        encoding_vectors = []

        for _ in range(self.M):
            x_l = self.Q_nn(x_l)
            b_i = custom_round(np.mod(np.dot(self.G_inv, x_l), self.q)).astype(int)
            encoding_vectors.append(b_i)
            x_l = x_l / self.q

        overload_error = not np.allclose(self.Q_nn(x_l), 0, atol=1e-8)
        return tuple(encoding_vectors), overload_error

    def encode(
        self, x: np.ndarray, with_dither: bool = False
    ) -> Tuple[Tuple[np.ndarray, ...], int]:
        """
        Encode a vector using hierarchical nested lattice quantization.

        This method quantizes the input vector using M hierarchical levels
        and handles overload by scaling the vector until quantization succeeds.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to be quantized.
        with_dither : bool, optional
            Whether to apply dithering during quantization. Default is False.

        Returns:
        --------
        Tuple[Tuple[numpy.ndarray, ...], int]
            (b_list, T) where b_list is a tuple of M encoding vectors and
            T is the number of scaling operations performed to handle overload.
        """
        b_list, did_overload = self._encode(x, with_dither)
        t = 0

        if self.overload:
            while did_overload and t < self.config.max_scaling_iterations:
                t += 1
                x = x / (2**self.alpha)
                b_list, did_overload = self._encode(x, with_dither)

            if did_overload:
                warnings.warn(
                    f"Overload not resolved after {self.config.max_scaling_iterations} iterations. "
                    "Consider increasing max_scaling_iterations or adjusting parameters."
                )
        else:
            b_list, did_overload = self._encode(x, with_dither)

        return b_list, t

    # ============================================================================
    # Decoding Methods
    # ============================================================================

    def _decode(self, b_list: Tuple[np.ndarray, ...], with_dither: bool) -> np.ndarray:
        """
        Internal decoding function that performs hierarchical reconstruction.

        This method reconstructs the original vector from M levels of
        encoding vectors using the hierarchical decoding algorithm.

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        with_dither : bool
            Whether dithering was applied during encoding.

        Returns:
        --------
        numpy.ndarray
            Reconstructed vector.
        """
        x_hat_list = []
        for b in b_list:
            # Compute quantization error directly
            Gb = np.dot(self.G, b)
            x_i_hat = Gb - self.q * self.Q_nn(Gb / self.q)
            x_hat_list.append(x_i_hat)

        x_hat = sum([np.power(self.q, i) * x_i for i, x_i in enumerate(x_hat_list)])

        if with_dither:
            x_hat = x_hat - self.dither

        return self.beta * x_hat

    def decode(
        self, b_list: Tuple[np.ndarray, ...], T: int, with_dither: bool = False
    ) -> np.ndarray:
        """
        Decode hierarchical encoding vectors back to the original space.

        This method reconstructs the original vector from its hierarchical
        encoding, accounting for any scaling that was applied during encoding.

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool, optional
            Whether dithering was applied during encoding. Default is False.

        Returns:
        --------
        numpy.ndarray
            Reconstructed vector.
        """
        return self._decode(b_list, with_dither) * (2 ** (self.alpha * T))

    def decode_coarse_to_fine(
        self,
        b_list: Tuple[np.ndarray, ...],
        T: int,
        with_dither: bool = False,
        depth: Optional[int] = None,
    ) -> np.ndarray:
        """
        Decode hierarchical encoding vectors with coarse-to-fine reconstruction.

        This method allows decoding from coarse to fine levels, where higher M
        means coarser quantization. The reconstruction can be stopped at any
        level from M-1 down to 0, providing progressive refinement.

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool, optional
            Whether dithering was applied during encoding. Default is False.
        depth : int, optional
            Maximum level to decode up to (M >= depth >= 1).
            If None, decodes all levels (equivalent to decode method).
            Higher depth means more detailed reconstruction.

        Returns:
        --------
        numpy.ndarray
            Reconstructed vector at the specified level of detail.
        """
        if depth is None:
            # Default behavior: decode all levels
            return self.decode(b_list, T, with_dither)

        if not (1 <= depth <= self.M):
            raise ValueError(f"depth must be between 1 and {self.M}, got {depth}")

        # Use the same reconstruction formula as _decode method for consistency
        x_hat_list = []
        for i in range(self.M):
            b = b_list[i]
            # Compute quantization error directly (same as _decode)
            Gb = np.dot(self.G, b)
            x_i_hat = Gb - self.q * self.Q_nn(Gb / self.q)
            x_hat_list.append(x_i_hat)

        # Sum backwards based on depth: 1 to M
        # x_hat = sum([np.power(self.q, i) * x_hat_list[i] for i in range(0, depth, 1)] )
        x_hat = sum(
            [
                np.power(self.q, i) * x_hat_list[i]
                for i in range(self.M - 1, self.M - 1 - depth, -1)
            ]
        )

        if with_dither:
            x_hat = x_hat - self.dither

        return self.beta * x_hat * (2 ** (self.alpha * T))

    def decode_progressive(
        self, b_list: Tuple[np.ndarray, ...], T: int, with_dither: bool = False
    ) -> List[np.ndarray]:
        """
        Generate progressive reconstructions from coarse to fine.

        This method returns a list of reconstructions at each level,
        from the coarsest (level 0) to the finest (level M-1).

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool, optional
            Whether dithering was applied during encoding. Default is False.

        Returns:
        --------
        List[numpy.ndarray]
            List of reconstructed vectors, from coarsest to finest.
        """
        reconstructions = []
        for level in range(1, self.M + 1):  # From 1 to M
            reconstruction = self.decode_coarse_to_fine(b_list, T, with_dither, level)
            reconstructions.append(reconstruction)
        return reconstructions

    def get_default_decoding(
        self,
        b_list: Tuple[np.ndarray, ...],
        T: int,
        with_dither: bool = False,
        depth: Optional[int] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get the default decoding based on the decoding parameter.

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool, optional
            Whether dithering was applied during encoding. Default is False.
        depth : int, optional
            Maximum level for coarse-to-fine decoding.

        Returns:
        --------
        numpy.ndarray or List[numpy.ndarray]
            Decoded result based on the decoding parameter.
        """
        if self.decoding == "full":
            return self.decode(b_list, T, with_dither)
        elif self.decoding == "coarse_to_fine":
            return self.decode_coarse_to_fine(b_list, T, with_dither, depth=depth)
        elif self.decoding == "progressive":
            return self.decode_progressive(b_list, T, with_dither)
        else:
            raise ValueError(f"Unknown decoding method: {self.decoding}")

    def decode_with_depth(
        self,
        b_list: Tuple[np.ndarray, ...],
        T: int,
        depth: int,
        with_dither: bool = False,
    ) -> np.ndarray:
        """
        Decode with a specific depth level.

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool, optional
            Whether dithering was applied during encoding. Default is False.
        depth : int
            Decoding depth (1 to M). Higher depth means finer reconstruction.

        Returns:
        --------
        numpy.ndarray
            Decoded result at the specified depth.
        """
        if depth < 1 or depth > self.M:
            raise ValueError(f"Depth must be between 1 and {self.M}, got {depth}")

        return self.decode_coarse_to_fine(b_list, T, with_dither, depth)

    # ============================================================================
    # Combined Methods
    # ============================================================================

    def quantize(self, x: np.ndarray, with_dither: bool = False) -> np.ndarray:
        """
        Complete hierarchical quantization process: encode and decode a vector.

        This is a convenience method that performs both encoding and decoding
        in a single call, returning the quantized version of the input vector.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to be quantized.
        with_dither : bool, optional
            Whether to apply dithering during quantization. Default is False.

        Returns:
        --------
        numpy.ndarray
            Quantized version of the input vector.
        """
        b_list, T = self.encode(x, with_dither)
        return self.decode(b_list, T, with_dither)

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def create_q_codebook(
        self, with_dither: bool = False
    ) -> Dict[Tuple[int, ...], np.ndarray]:
        """
        Create a codebook for the hierarchical quantizer.

        This method creates a codebook by using the nested lattice quantizer
        with appropriate parameters. The codebook maps encoding vectors to
        their corresponding lattice points.

        Parameters:
        -----------
        with_dither : bool, optional
            Whether to apply dithering when creating the codebook. Default is False.

        Returns:
        --------
        Dict[Tuple[int, ...], numpy.ndarray]
            Dictionary mapping encoding vectors (as tuples) to lattice points.

        Notes:
        ------
        The codebook is created using the nested lattice quantizer with
        the same parameters as this hierarchical quantizer, providing
        a mapping for lookup-based operations.
        """
        # Create NLQ instance with the same configuration but without M parameter
        nlq_config = {
            "lattice_type": self.config.lattice_type,
            "q": self.config.q,
            "beta": self.config.beta,
            "alpha": self.config.alpha,
            "eps": self.config.eps,
            "overload": self.config.overload,
            "max_scaling_iterations": self.config.max_scaling_iterations,
            "with_tie_dither": self.config.with_tie_dither,
            "with_dither": self.config.with_dither,
        }
        nq = NQ(nlq_config, self.G, self.Q_nn)
        return nq.create_codebook(with_dither)

    def get_rate_distortion_info(self) -> Dict[str, Any]:
        """
        Get information about the rate-distortion characteristics of the quantizer.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing rate-distortion information.
        """
        return {
            "lattice_type": self.lattice_type,
            "q": self.q,
            "M": self.M,
            "beta": self.beta,
            "alpha": self.alpha,
            "overload": self.overload,
            "decoding": self.decoding,
            "max_scaling_iterations": self.config.max_scaling_iterations,
            "with_tie_dither": self.with_tie_dither,
            "with_dither": self.with_dither,
        }

    def get_config(self) -> HNLQConfig:
        """
        Get the current configuration of the quantizer.

        Returns:
        --------
        HNLQConfig
            Current configuration.
        """
        return self.config

    def update_config(self, new_config: HNLQConfig) -> None:
        """
        Update the quantizer configuration.

        Parameters:
        -----------
        new_config : HNLQConfig
            New configuration to apply.
        """
        self.config = new_config

    def validate_config(self) -> None:
        """Validate the current configuration."""
        self.config.__post_init__()

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "lattice_type": self.lattice_type,
            "lattice_dimension": self._dim,
            "quantization_parameter": self.q,
            "hierarchical_levels": self.M,
            "scaling_parameters": {"beta": self.beta, "alpha": self.alpha},
            "perturbation": self.eps,
            "overload_handling": self.overload,
            "decoding_method": self.decoding,
            "max_iterations": self.config.max_scaling_iterations,
            "dither_settings": {
                "with_tie_dither": self.with_tie_dither,
                "with_dither": self.with_dither,
            },
        }

    def __repr__(self) -> str:
        """String representation of the quantizer."""
        return (
            f"HNLQ(lattice={self.lattice_type}, dim={self._dim}, q={self.q}, "
            f"M={self.M}, beta={self.beta:.3f}, alpha={self.alpha:.3f}, "
            f"overload={self.overload}, decoding={self.decoding})"
        )

    def __str__(self) -> str:
        """String representation of the quantizer."""
        return self.__repr__()

    def batch_encode(
        self, X: np.ndarray, with_dither: bool = False
    ) -> Tuple[List[Tuple[np.ndarray, ...]], List[int]]:
        """
        Encode multiple vectors efficiently using hierarchical nested lattice quantization.

        Parameters:
        -----------
        X : numpy.ndarray
            Input matrix where each row is a vector to encode.
        with_dither : bool, optional
            Whether to apply dithering during quantization. Default is False.

        Returns:
        --------
        tuple
            (encoded_vectors, scaling_counts) where encoded_vectors is a list
            of tuples of M encoding vectors and scaling_counts is a list of scaling counts.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        encoded_vectors = []
        scaling_counts = []

        for i in range(X.shape[0]):
            b_list, T = self.encode(X[i], with_dither)
            encoded_vectors.append(b_list)
            scaling_counts.append(T)

        return encoded_vectors, scaling_counts

    def batch_decode(
        self,
        encoded_vectors: List[Tuple[np.ndarray, ...]],
        scaling_counts: List[int],
        with_dither: bool = False,
        decoding: str = "full",
    ) -> np.ndarray:
        """
        Decode multiple vectors efficiently using hierarchical nested lattice quantization.

        Parameters:
        -----------
        encoded_vectors : list
            List of tuples of M encoding vectors.
        scaling_counts : list
            List of scaling counts corresponding to each encoded vector.
        with_dither : bool, optional
            Whether dithering was applied during encoding. Default is False.
        decoding : str, optional
            Decoding method to use ('full', 'coarse_to_fine', 'progressive').
            Default is "full".

        Returns:
        --------
        numpy.ndarray
            Matrix where each row is a decoded vector.
        """
        if len(encoded_vectors) != len(scaling_counts):
            raise ValueError(
                "Number of encoded vectors must match number of scaling counts"
            )

        decoded_vectors = []
        for b_list, T in zip(encoded_vectors, scaling_counts):
            if decoding == "full":
                decoded = self.decode(b_list, T, with_dither)
            elif decoding == "coarse_to_fine":
                decoded = self.decode_coarse_to_fine(b_list, T, with_dither)
            elif decoding == "progressive":
                # For progressive, we take the finest level (last element)
                progressive_results = self.decode_progressive(b_list, T, with_dither)
                decoded = progressive_results[-1]  # Take the finest reconstruction
            else:
                raise ValueError(f"Unknown decoding method: {decoding}")

            decoded_vectors.append(decoded)

        return np.array(decoded_vectors)

    # ============================================================================
    # Class Methods for Creating Quantizers
    # ============================================================================

    @classmethod
    def create_z2_quantizer(
        cls,
        q: int,
        M: int,
        beta: float = 1.0,
        alpha: float = 1.0,
        eps: float = 1e-8,
        overload: bool = True,
        decoding: str = "full",
        with_tie_dither: bool = True,
        with_dither: bool = False,
    ) -> "HNLQ":
        """
        Create a hierarchical quantizer for the Z² lattice (identity matrix).

        Parameters:
        -----------
        q : int
            Quantization parameter.
        M : int
            Number of hierarchical levels.
        beta : float, optional
            Scaling parameter. Default is 1.0.
        alpha : float, optional
            Scaling parameter for overload handling. Default is 1.0.
        eps : float, optional
            Small perturbation parameter. Default is 1e-8.
        overload : bool, optional
            Whether to handle overload by scaling. Default is True.
        decoding : str, optional
            Default decoding method. Default is "full".
        with_tie_dither : bool, optional
            Whether to add tie dither. Default is True.
        with_dither : bool, optional
            Whether to add dither. Default is False.

        Returns:
        --------
        HNLQ
            Configured hierarchical quantizer for Z² lattice.
        """
        config = HNLQConfig(
            lattice_type="Z2",
            q=q,
            M=M,
            beta=beta,
            alpha=alpha,
            eps=eps,
            overload=overload,
            decoding=decoding,
            max_scaling_iterations=10,
            with_tie_dither=with_tie_dither,
            with_dither=with_dither,
        )
        return cls(config)

    @classmethod
    def create_d4_quantizer(
        cls,
        q: int,
        M: int,
        beta: float = 1.0,
        alpha: float = 1.0,
        eps: float = 1e-8,
        overload: bool = True,
        decoding: str = "full",
        with_tie_dither: bool = True,
        with_dither: bool = False,
    ) -> "HNLQ":
        """
        Create a hierarchical quantizer for the D₄ lattice.

        Parameters:
        -----------
        q : int
            Quantization parameter.
        M : int
            Number of hierarchical levels.
        beta : float, optional
            Scaling parameter. Default is 1.0.
        alpha : float, optional
            Scaling parameter for overload handling. Default is 1.0.
        eps : float, optional
            Small perturbation parameter. Default is 1e-8.
        overload : bool, optional
            Whether to handle overload by scaling. Default is True.
        decoding : str, optional
            Default decoding method. Default is "full".
        with_tie_dither : bool, optional
            Whether to add tie dither. Default is True.
        with_dither : bool, optional
            Whether to add dither. Default is False.

        Returns:
        --------
        HNLQ
            Configured hierarchical quantizer for D₄ lattice.
        """
        config = HNLQConfig(
            lattice_type="D4",
            q=q,
            M=M,
            beta=beta,
            alpha=alpha,
            eps=eps,
            overload=overload,
            decoding=decoding,
            max_scaling_iterations=10,
            with_tie_dither=with_tie_dither,
            with_dither=with_dither,
        )
        return cls(config)

    @classmethod
    def create_e8_quantizer(
        cls,
        q: int,
        M: int,
        beta: float = 1.0,
        alpha: float = 1.0,
        eps: float = 1e-8,
        overload: bool = True,
        decoding: str = "full",
        with_tie_dither: bool = True,
        with_dither: bool = False,
    ) -> "HNLQ":
        """
        Create a hierarchical quantizer for the E₈ lattice.

        Parameters:
        -----------
        q : int
            Quantization parameter.
        M : int
            Number of hierarchical levels.
        beta : float, optional
            Scaling parameter. Default is 1.0.
        alpha : float, optional
            Scaling parameter for overload handling. Default is 1.0.
        eps : float, optional
            Small perturbation parameter. Default is 1e-8.
        overload : bool, optional
            Whether to handle overload by scaling. Default is True.
        decoding : str, optional
            Default decoding method. Default is "full".
        with_tie_dither : bool, optional
            Whether to add tie dither. Default is True.
        with_dither : bool, optional
            Whether to add dither. Default is False.

        Returns:
        --------
        HNLQ
            Configured hierarchical quantizer for E₈ lattice.
        """
        config = HNLQConfig(
            lattice_type="E8",
            q=q,
            M=M,
            beta=beta,
            alpha=alpha,
            eps=eps,
            overload=overload,
            decoding=decoding,
            max_scaling_iterations=10,
            with_tie_dither=with_tie_dither,
            with_dither=with_dither,
        )
        return cls(config)
