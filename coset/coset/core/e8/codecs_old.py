"""
E8 Lattice Codecs Module

This module provides optimized encode/decode functions specifically for E8 lattice
quantization, including both CPU and GPU implementations.
"""

import warnings
from typing import Tuple, Optional
import torch

from ..base import LatticeConfig, Lattice
from .lattice import E8Lattice
from enum import Enum

#######################################################original##################################################
# def e8_encode(
#     x: torch.Tensor, 
#     config: LatticeConfig,
#     lattice: Optional[Lattice] = None,
#     dither: Optional[torch.Tensor] = None,
#     device: Optional[torch.device] = None,
# ) -> Tuple[torch.Tensor, int]:
#     """
#     E8 hierarchical encoding.
    
#     Encode vectors using hierarchical nested lattice quantization with M levels
#     and handle overload by scaling the vector until quantization succeeds.
    
#     Args:
#         x: Input vector(s) to be encoded (shape [d] or [batch_size, d])
#         config: LatticeConfig configuration
#         lattice: Optional lattice instance (defaults to E8Lattice)
#         dither: Optional dither vector for randomized quantization
#         device: Device to perform computation on (defaults to x's device)
        
#     Returns:
#         Tuple of (encoding_vectors, T) where:
#         - encoding_vectors: Tensor of shape [batch_size, M, d] or [M, d] containing M encoding vectors
#         - T: Number of scaling operations performed to handle overload
#     """
#     # Determine device
#     if device is None:
#         device = x.device
    
#     if lattice is None:
#         lattice = E8Lattice(device=device)
    
#     # Handle both single vector and batch cases
#     if x.dim() == 1:
#         x = x.unsqueeze(0)  # Add batch dimension
#         squeeze_output = True
#     else:
#         squeeze_output = False
    
#     # Ensure x is on the correct device
#     x = x.to(device)
#     batch_size, d = x.shape
#     if d != 8:
#         raise ValueError(f"Input dimension {d} doesn't match E8 dimension 8")
        
    
#     # Apply scaling and dithering
#     x_scaled = x / config.beta
#     if config.with_dither and dither is not None:
#         dither = dither.to(device)
#         if dither.dim() == 1:
#             dither = dither.unsqueeze(0)  # Add batch dimension
#         x_scaled = x_scaled + dither
    
#     # Perform hierarchical encoding
#     x_l = x_scaled.clone()
#     encoding_vectors = []
    
#     for _ in range(config.M):
#         # encode to E8 lattice (vectorized for batch)
#         encoded = lattice.projection(x_l)  # [batch_size, 8] --> nearest E8 lattice point
#         encoded = lattice.encode_coords(encoded, config.q)  # [batch_size, 8] ---> take that lattice point and express it in lattice coordinates--> radix-q digits
#         encoding_vectors.append(encoded)
#         # Scale down for next level
#         x_l = x_l / config.q
    
#     encoding_vectors = torch.stack(encoding_vectors, dim=1)  # [batch_size, M, 8]
#     #################pass through projection, and lastly it should be 0#######################################
#     # Remove batch dimension if input was single vector
#     if squeeze_output:
#         encoding_vectors = encoding_vectors.squeeze(0)  # [M, 8]
    
#     return encoding_vectors, 0  # No scaling iterations needed
###############################################################################################################################
def e8_encode(
    x: torch.Tensor, 
    config: LatticeConfig,
    lattice: Optional[Lattice] = None,
    dither: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) ->  Tuple[torch.Tensor, torch.Tensor, int]:
    """
    E8 hierarchical encoding.
    
    Encode vectors using hierarchical nested lattice quantization with M levels
    and handle overload by scaling the vector until quantization succeeds.
    
    Args:
        x: Input vector(s) to be encoded (shape [d] or [batch_size, d])
        config: LatticeConfig configuration
        lattice: Optional lattice instance (defaults to E8Lattice)
        dither: Optional dither vector for randomized quantization
        device: Device to perform computation on (defaults to x's device)
        
    Returns:
        Tuple of (encoding_vectors, T) where:
        - encoding_vectors: Tensor of shape [batch_size, M, d] or [M, d] containing M encoding vectors
        - T: Number of scaling operations performed to handle overload
    """
    # Determine device
    if device is None:
        device = x.device
    
    if lattice is None:
        lattice = E8Lattice(device=device)
    
    # Handle both single vector and batch cases
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Ensure x is on the correct device
    x = x.to(device)
    batch_size, d = x.shape
    if d != 8:
        raise ValueError(f"Input dimension {d} doesn't match E8 dimension 8")
        
    
    # Apply scaling and dithering
    x_scaled = x / config.beta
    if config.with_dither and dither is not None:
        dither = dither.to(device)
        if dither.dim() == 1:
            dither = dither.unsqueeze(0)  # Add batch dimension
        x_scaled = x_scaled + dither
    
    # Perform hierarchical encoding
    x_l = x_scaled.clone()
    encoding_vectors = []
    
    for _ in range(config.M):
        # encode to E8 lattice (vectorized for batch)
        encoded = lattice.projection(x_l)  # [batch_size, 8] --> nearest E8 lattice point
        encoded = lattice.encode_coords(encoded, config.q)  # [batch_size, 8] ---> take that lattice point and express it in lattice coordinates--> radix-q digits
        encoding_vectors.append(encoded)
        # Scale down for next level
        x_l = x_l / config.q
    
    encoding_vectors = torch.stack(encoding_vectors, dim=1)  # [batch_size, M, 8]
    #################pass through projection, and lastly it should be 0#######################################
    residual_proj = lattice.projection(x_l)
    tol = 1e-8
    overload_flags = residual_proj.norm(dim=1) > tol
    # Remove batch dimension if input was single vector
    if squeeze_output:
        encoding_vectors = encoding_vectors.squeeze(0)  # [M, 8]
        overload_flags = overload_flags.squeeze(0)
    return encoding_vectors, overload_flags, 0  # No scaling iterations needed

class DecodingMethod(Enum):
    """Enumeration of available decoding methods."""
    
    FULL = "full"
    # APPROXIMATE and PROGRESSIVE will be added later


class E8Decoder:
    """
    E8 lattice decoder with multiple decoding methods.
    
    This class provides different decoding strategies for E8 lattice quantization,
    allowing users to choose between accuracy and speed based on their needs.
    
    Methods:
    - full: Complete hierarchical decoding (default, most accurate)
    - approximate: Quick approximation for real-time applications (to be added)
    - progressive: Incremental decoding with intermediate results (to be added)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize E8 decoder.
        
        Args:
            device: Device to perform computation on (defaults to CPU)
        """
        self.device = device if device is not None else torch.device('cpu')
        self.lattice = E8Lattice(device=self.device)
    
    def decode(self, b: torch.Tensor, config: LatticeConfig, method: DecodingMethod = DecodingMethod.FULL) -> torch.Tensor:
        """
        Decode E8 lattice encoding vectors.
        
        Args:
            b: Encoding vectors of shape [batch_size, M, 8] or [M, 8]
            config: Lattice quantization configuration
            method: Decoding method to use (default: FULL)
            
        Returns:
            Decoded tensor of shape [batch_size, 8] or [8]
        """
        if method == DecodingMethod.FULL:
            return self._full_decode(b, config)
        else:
            raise ValueError(f"Unknown decoding method: {method}. Only 'full' is currently supported.")
    
    def _full_decode(self, b: torch.Tensor, config: LatticeConfig) -> torch.Tensor:
        """
        Full hierarchical decoding (most accurate).
        
        Performs complete hierarchical reconstruction with all levels.
        """
        # Handle both single vector and batch cases
        if b.dim() == 2:
            b = b.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, M, d = b.shape
        device = b.device
        
        # Ensure lattice is on correct device
        if self.lattice.device != device:
            self.lattice = E8Lattice(device=device)
        
        # Hierarchical reconstruction
        x_hat_list = []
        for i in range(M):
            b_i = b[:, i, :]  # [batch_size, 8]
            # Convert encoding coordinates to lattice point
            Gb = self.lattice.decode_coords(b_i, config.q)
            # Compute quantization error
            x_i_hat = Gb - config.q * self.lattice.projection(Gb / config.q) #projections ---> x_i_hat is reconstructed vector
            x_hat_list.append(x_i_hat)
        
        # Sum with appropriate weights
        x_hat = torch.zeros_like(x_hat_list[0])
        for i, x_i in enumerate(x_hat_list):
            x_hat += (config.q ** i) * x_i
        
        # Apply scaling compensation
        x_hat = x_hat * config.beta
        
        if squeeze_output:
            x_hat = x_hat.squeeze(0)
        
        return x_hat
    

def e8_decode(
    b: torch.Tensor,
    config: LatticeConfig,
    method: str = "full",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    E8 hierarchical decoding wrapper function.
    
    This function provides a simple interface to the E8Decoder class,
    allowing users to choose different decoding methods.
    
    Args:
        b: Encoding vectors of shape [batch_size, M, 8] or [M, 8]
        config: Lattice quantization configuration
        method: Decoding method ("full" - only method currently supported)
        device: Device to perform computation on (defaults to b's device)
        
    Returns:
        Decoded tensor of shape [batch_size, 8] or [8]
    """
    # Convert string method to enum
    method_map = {
        "full": DecodingMethod.FULL,
        # "approximate": DecodingMethod.APPROXIMATE,  # To be added later
        # "progressive": DecodingMethod.PROGRESSIVE,  # To be added later
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown decoding method: {method}. Available: {list(method_map.keys())}")
    
    # Create decoder and decode
    decoder = E8Decoder(device=device)
    return decoder.decode(b, config, method_map[method])


def e8_quantize(x: torch.Tensor, q: int, lattice: Optional[E8Lattice] = None) -> torch.Tensor:
    """
    E8 quantization by combining encoding and decoding.
    
    This function first encodes the input using hierarchical nested lattice quantization,
    then decodes it back to get a valid E8 lattice point.
    
    Args:
        x: Input tensor to quantize (shape [8] or [batch_size, 8])
        q: Quantization parameter
        lattice: Optional E8Lattice instance (creates new one if None)
        
    Returns:
        Quantized tensor (valid E8 lattice point)
    """
    if lattice is None:
        lattice = E8Lattice(device=x.device)
    
    # Handle batch inputs - use vectorized processing instead of Python loop
    if x.dim() == 2:
        # Batch processing with vectorized operations
        batch_size = x.shape[0]
        
        # Create a simple config for quantization
        config = LatticeConfig(
            lattice_type='E8',
            q=q,
            M=2,  # Default M=2 for hierarchical quantization
            beta=1.0,  # No additional scaling
            alpha=1.0,
            max_scaling_iterations=10,
            with_dither=False,
            disable_overload_protection=True
        )
        
        # Step 1: vectorized encoding for all vectors at once
        encoding_vectors, t = e8_encode(x, config, lattice=lattice)
        
        # Step 2: vectorized decoding for all vectors at once
        x_final = e8_decode(encoding_vectors, config, method="full")
        
        return x_final
    
    # Single vector processing
    if x.shape[0] != 8:
        raise ValueError(f"Input dimension {x.shape[0]} doesn't match E8 dimension 8")
    
    # Create a simple config for quantization
    config = LatticeConfig(
        lattice_type='E8',
        q=q,
        M=2,  # Default M=2 for hierarchical quantization
        beta=1.0,  # No additional scaling
        alpha=1.0,
        max_scaling_iterations=10,
        with_dither=False,
        disable_overload_protection=True
    )
    
    # Step 1: encoding
    encoding_vectors, t = e8_encode(x, config, lattice=lattice)
    
    # Step 2: decoding
    x_final = e8_decode(encoding_vectors, config, method="full")
    
    return x_final


def e8_quantize_fused_debug(
    x: torch.Tensor,
    config: LatticeConfig,
    lattice: Optional[E8Lattice] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused hierarchical quantizer following the supervisor pseudocode.

    Implements a single forward loop that accumulates the per-level residuals:
        g_tilde ← x
        x_hat ← 0
        for m = 0 .. M-1:
            g_tilde ← Q_L(g_tilde)
            layer_vec ← g_tilde − q · Q_L(g_tilde / q)
            x_hat ← x_hat + (q^m) · layer_vec
            g_tilde ← g_tilde / q
        overload_error ← 1 if Q_L(g_tilde) ≠ 0 else 0

    Args:
        x: Input tensor of shape [8] or [batch_size, 8]
        config: LatticeConfig containing q, M, beta, etc.
        lattice: Optional E8Lattice instance (creates new one if None)
        device: Optional device override

    Returns:
        Tuple (x_hat, overload_flag) where:
            x_hat: Quantized tensor matching input shape
            overload_flag: Tensor of dtype bool indicating overload per sample
    """
    if device is None:
        device = x.device

    if lattice is None:
        lattice = E8Lattice(device=device)

    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    x = x.to(device)

    if x.size(-1) != lattice.d:
        raise ValueError(f"Input dimension {x.size(-1)} doesn't match E8 dimension {lattice.d}")

    q_tensor = torch.as_tensor(config.q, dtype=x.dtype, device=device)
    g_tilde = x/config.beta
    #x=x/config.beta #normalize the input
    x_hat = torch.zeros_like(g_tilde)

    for m in range(config.M):
        g_tilde = lattice.projection(g_tilde)
        coarse = lattice.projection(g_tilde / q_tensor)
        layer_vec = g_tilde - q_tensor * coarse
        x_hat = x_hat + (q_tensor.pow(m)) * layer_vec
        g_tilde = g_tilde / q_tensor

    if config.check_overload:
        overload_proj = lattice.projection(g_tilde)
        overload_flag = overload_proj.abs().amax(dim=-1) > 0.0
    else:
        overload_flag = torch.zeros(g_tilde.shape[0], dtype=torch.bool, device=g_tilde.device)

    if squeeze_output:
        x_hat = x_hat.squeeze(0)
        overload_flag = overload_flag.squeeze(0)

    return x_hat, overload_flag

# def e8_quantize_fused(
#     x: torch.Tensor,
#     config: LatticeConfig,
#     lattice: Optional[E8Lattice] = None,
#     device: Optional[torch.device] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
    
	
#     if lattice is None:
#         lattice = E8Lattice(device=x.device)
#     #x=x/config.beta
#     return lattice.projection(x)
# Correct return type: a single tensor (you were returning just one anyway)
def e8_quantize_fused(
    x: torch.Tensor,
    config: LatticeConfig,
    lattice: Optional[E8Lattice] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Project 8D blocks to nearest E8 point with *proper step scaling* derived from (q, M, Delta0).
    This function intentionally ignores config.beta; keep beta handling in the layer.
    """
    if device is None:
        device = x.device
    if lattice is None:
        lattice = E8Lattice(device=device)

    # --- 1) compute the lattice step for the deepest level ---
    # Prefer: Delta0 provided in config (if your config has it)
    # Otherwise ask the lattice to compute one compatible with your implementation.
    # If your LatticeConfig does not carry Delta0, pass it in via the adapter and close over it.
    if hasattr(config, "Delta0") and config.Delta0 is not None:
        Delta0 = float(config.Delta0)
    else:
        # falls back to geometric default; adjust rho to what HNLQLinearQAT uses
        # (0.95 was common in your code)
        Delta0 = float(lattice.compute_delta0(config.q, config.M, rho=0.95))

    step = Delta0 / (config.q ** config.M)  # base grid at the deepest level

    # --- 2) reshape to [N, 8] (E8 blocks), keep contiguous to satisfy view() ---
    x = x.contiguous().view(-1, 8)

    # --- 3) scale into lattice units, project, scale back ---
    z = x / step
    zq = lattice.projection(z)          # nearest E8 point in lattice units
    y = zq * step                       # back to model units

    return y

# def _exact_fused_hierarchical(x_scaled, q, M, lattice): #excat
#     acc = torch.zeros_like(x_scaled)
#     x_l = x_scaled
#     for m in range(M): #remove loop
#         z_m = lattice.projection(x_l)             # ← no / (q**m) here
#         #r = z_m - q * lattice.projection(z_m / q)
#         #acc = acc + (q**m) * r
#         x_l = x_l / q
#     return acc


# def e8_quantize_fused_exact_loop(
#     x: torch.Tensor,
#     q: int,			#radix
#     M: int = 2,		# no. of hierarchical digits/levels
#     beta: float = 1.0,
#     lattice: Optional[E8Lattice] = None,
#     mode: str = "codec_compat",  # "codec_compat" or "projector_safe"
# ) -> torch.Tensor:
#     if lattice is None:
#         lattice = E8Lattice(device=x.device)
#     squeeze = (x.dim() == 1)
#     if squeeze: x = x.unsqueeze(0)    #if x is [8], make it [1,8] so the code can be batch-vectorized
#     if x.size(-1) != 8:
#         raise ValueError(f"E8 expects dim=8, got {x.size(-1)}")

#     qf = torch.as_tensor(q, dtype=x.dtype, device=x.device)
#     x_l = x / beta					#normalized domain
#     acc = torch.zeros_like(x_l)    	#Accumulator for the hierarchical sum

#     for m in range(M):
#         z = lattice.projection(x_l)	#Exact nearest-neighbor projection to E8--> lattice point
#         '''codec_compat reproduces existing encode_coords/decode_coords behavior (round in coord space + non-centered modulo). That guarantees bit-exact 
#         parity with legacy results on the first pass. 
#         But re-applying can select a different coset representative (because modulo wraps), so it's not 
#         idempotent.'''
#         '''projector_safe computes the residual directly from the canonical nearest lattice point at each level 
#         (no coord modulo). That makes it 
#         idempotent on any input that’s representable with M digits (i.e., not overloaded).'''
#         if mode == "codec_compat":
#             # Match the current encode_coords/decode_coords pipeline
#             # (non-centered modulo + round in coord space)
#             GinvT = lattice.G_inv.T		#converts real-space lattice points to coordinate space.
#             GT    = lattice.G.T			#maps coordinates back to real space.
#             c     = torch.round(z @ GinvT)	#Convert z to integer coordinates
#             # modulo into [0, q-1]
#             c_mod = torch.remainder(c, qf)	#Force each coordinate digit into the non-centered range[0, q-1]
#             Gb    = c_mod @ GT			#Map those digits back to a representative lattice point Gb
#             corr  = lattice.projection(Gb / qf)	#Compute the coarse component at the next level
#             r     = Gb - qf * corr			#The level-m residual in real space
	    
#         elif mode == "projector_safe":
#             # Idempotent exact residual (no coord modulo)
#             corr = lattice.projection(z / qf)	#Use the actual projected point z to compute the exact residual.
#             r    = z - qf * corr            # r=z-q Q(z/q)

#         else:
#             raise ValueError("mode must be 'codec_compat' or 'projector_safe'")

#         acc = acc + (qf ** m) * r		#Accumulate this level’s residual weighted by q^m
#         x_l = x_l / qf				#Prepare the next level’s input

#     x_hat = acc * beta				#Rescale back to the original domain
#     return x_hat.squeeze(0) if squeeze else x_hat



# # ---- Overload detector (practical) ----
# def is_overloaded(x: torch.Tensor, q: int, M: int, beta: float, lattice: E8Lattice, eps: float = 1e-6) -> bool:
#     """
#     Heuristic: run the same level-splitting M times (like fused),
#     then see if there is *tail energy* that would need an (M+1)th digit.
#     If Π(x_l) after M levels is nonzero, we call it overloaded.
#     """
#     x_l = x / beta
#     qf  = torch.as_tensor(q, dtype=x.dtype, device=x.device)

#     for _ in range(M):
#         x_l = x_l / qf  # match the fused loop's downscaling

#     tail = lattice.projection(x_l)  # what the (M+1)th level would quantize
#     return (tail.abs().max().item() > eps)    
# def e8_quantize_fused_babai(x: torch.Tensor, q: int, lattice: Optional[E8Lattice] = None) -> torch.Tensor:
#     """
#     Fused hierarchical quantization using babai approximation E8 projection (Eq. 9★ residual form).
    
#     Implements the fused equation:
#         x̂ = β * Σ_{m=0}^{M-1} q^m * [Babai(x/(β q^m)) - q * Babai(Babai(x/(β q^m))/q)]
    
#     Args:
#         x: Input tensor of shape [8] or [batch_size, 8]
#         config: LatticeConfig containing beta, q, M, etc.
#         lattice: Optional E8Lattice instance (creates new one if None)
        
#     Returns:
#         Quantized tensor of same shape as input
#     """
#     if lattice is None:
#         lattice = E8Lattice(device=x.device)
    
#     squeeze = False
#     if x.dim() == 2:
#         config = LatticeConfig(lattice_type='E8', q=q, M=2, beta=1.0, alpha=1.0,
#                             max_scaling_iterations=10, with_dither=False,
#                             disable_overload_protection=True)
#     x_scaled=x/config.beta
#     acc=torch.zeros_like(x_scaled)
#     x_l=x_scaled
#     for m in range(config.M):
#         z_m=lattice.babai_projection(x_l)
#         #r=z_m-config.q*lattice.babai_projection(z_m/config.q)
#         #acc=acc+(config.q**m)*r
#         x_l=x_l/config.q
#     return z_m#*config.beta


# class FusedEq9STE(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, q, M, lattice, beta: float, use_babai: bool):
#         if use_babai:
#              y = e8_quantize_fused_babai(x, q, M, lattice, beta)
#         else:
#             cfg = LatticeConfig(lattice_type='E8', q=q, M=M, beta=beta, alpha=1.0,
#                             max_scaling_iterations=0, with_dither=False,
#                             disable_overload_protection=True)
#         y = e8_quantize_fused_exact(x, cfg, lattice)
#         ctx.save_for_backward(x)
#         return y
#     @staticmethod
#     def backward(ctx, grad_out):
#         (x,) = ctx.saved_tensors
#         return grad_out.clone(), None, None, None, None, None

# def fused_eq9_ste(x, q, M, lattice, beta: float = 1.0, use_babai: bool = False):
#     return FusedEq9STE.apply(x, q, M, lattice, beta, use_babai)

