import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ----------- Straight-Through Estimators -----------
def ste_round(x): return (x - x.detach()) + x.detach().round()
def ste_clip(x, lo, hi): return x + (x.clamp(lo, hi) - x).detach()

# ----------- LSQ-A Activation Quantizer -----------
class LSQActivation(nn.Module):
    def __init__(self, bit_width=8, init_alpha=6.0):
        super().__init__()
        self.bit_width = bit_width
        self.qmin = -(2 ** (bit_width - 1))
        self.qmax = (2 ** (bit_width - 1)) - 1
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        alpha = torch.relu(self.alpha) + 1e-8
        x_clipped = ste_clip(x, -alpha, alpha)
        scale = alpha / self.qmax
        y = ste_round(x_clipped / scale)
        return y * scale

# ----------- Export Bundle Dataclass -----------
@dataclass
class HNLQExportBundle:
    weight_q: torch.Tensor
    bias_f: torch.Tensor | None
    meta: dict

# ----------- Main HNLQ Linear QAT-Lite Layer -----------
class HNLQLinear(nn.Module):
    """
    QAT-Lite Linear with flexible tiling and LSQ-A activation quantization.
    Biases remain float; weights quantized via HNLQ.
    """
    def __init__(self, in_dim, out_dim, G, Ginv, Delta0,
                 q=8, M=4, eta=0.2, k=5,
                 tiling='row', block_size=8,
                 use_bias=True, act_bit_width=8, act_init_alpha=6.0):
        super().__init__()
        assert tiling in ('row','block')
        assert in_dim % block_size == 0
        self.in_dim, self.out_dim = in_dim, out_dim
        self.q, self.M = q, M
        self.eta, self.k = eta, k
        self.Delta0 = float(Delta0)
        self.block_size, self.tiling = block_size, tiling

        # Weights and biases
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim).normal_(0, 0.02))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if use_bias else None
        self.s_row = nn.Parameter(torch.ones(out_dim))

        # Lattice buffers
        self.register_buffer('G', G.clone().detach())
        self.register_buffer('Ginv', Ginv.clone().detach())
        self.register_buffer('gamma_inv', (Ginv.abs().sum(dim=1)).max())

        # Tiling configuration
        tiles = out_dim if tiling == 'row' else out_dim * (in_dim // block_size)
        self.theta_beta = nn.Parameter(torch.zeros(tiles))
        self.register_buffer('sigma_ema', torch.ones(tiles))
        self.register_buffer('xmax_ema', torch.ones(tiles))
        self.ema_momentum = 0.99
        self.stat_update_interval = 256
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

        # Activation quantizer
        self.actq = LSQActivation(bit_width=act_bit_width, init_alpha=act_init_alpha)

    def _view_blocks(self, W):
        B = self.block_size
        return W.view(self.out_dim, self.in_dim // B, B)

    def _gather_stats(self, W_blocks):
        with torch.no_grad():
            m = self.ema_momentum
            if self.tiling == 'row':
                sigma = W_blocks.std(dim=(1,2)) + 1e-8
                xmax = W_blocks.abs().amax(dim=(1,2)) + 1e-8
                self.sigma_ema[:self.out_dim].mul_(m).add_((1-m)*sigma)
                self.xmax_ema[:self.out_dim].mul_(m).add_((1-m)*xmax)
            else:
                sigma = W_blocks.std(dim=2).reshape(-1) + 1e-8
                xmax = W_blocks.abs().amax(dim=2).reshape(-1) + 1e-8
                self.sigma_ema.mul_(m).add_((1-m)*sigma)
                self.xmax_ema.mul_(m).add_((1-m)*xmax)

    def _bounds(self):
        qM = float(self.q ** self.M)
        ginv = float(self.gamma_inv)
        beta_min = (self.Delta0 / qM) / (self.eta * ginv * self.sigma_ema)
        beta_max_det  = (self.Delta0 * qM) / (2 * ginv * self.xmax_ema)
        beta_max_prob = (self.Delta0 * qM) / (2 * self.k * ginv * self.sigma_ema)
        beta_max = torch.minimum(beta_max_det, beta_max_prob)
        beta_min = torch.minimum(beta_min, beta_max * 0.9)
        return beta_min, beta_max

    def _quantize_weights(self, W):
        B = self.block_size
        W_blocks = self._view_blocks(W)
        self.step_count += 1
        if (self.step_count % self.stat_update_interval) == 0:
            self._gather_stats(W_blocks.detach())

        beta_min, beta_max = self._bounds()

        if self.tiling == 'row':
            theta = self.theta_beta[:self.out_dim]
            beta_row = beta_min[:self.out_dim] + torch.sigmoid(theta) * (beta_max[:self.out_dim]-beta_min[:self.out_dim])
            beta = beta_row.view(-1,1,1)
        else:
            theta = self.theta_beta
            beta = beta_min + torch.sigmoid(theta)*(beta_max-beta_min)
            beta = beta.view(self.out_dim, self.in_dim // B, 1)

        B_scaled = W_blocks * beta
        Y = torch.einsum('ab,obk->oak', self.Ginv, B_scaled)
        step = self.Delta0 * (self.q ** (-self.M))
        Yq = ste_round(Y / step) * step
        Bq = torch.einsum('ab,oak->obk', self.G, Yq)
        return Bq.reshape_as(W)

    def forward(self, x):
        xq = self.actq(x)
        W = self.weight / (self.weight.norm(dim=1, keepdim=True)+1e-6)
        W = W * self.s_row.unsqueeze(1)
        Wq = self._quantize_weights(W)
        return F.linear(xq, Wq, self.bias)

    # -------- Export Helpers --------
    @torch.no_grad()
    def export_quantized(self):
        W = self.weight / (self.weight.norm(dim=1, keepdim=True)+1e-6)
        W = W * self.s_row.unsqueeze(1)
        Wq = self._quantize_weights(W)
        meta = dict(
            q=int(self.q), M=int(self.M), Delta0=float(self.Delta0),
            block_size=int(self.block_size), tiling=self.tiling,
            gamma_inv=float(self.gamma_inv),
            act_bit_width=int(self.actq.bit_width),
            act_alpha=float(torch.relu(self.actq.alpha).item()),
            ema={'sigma': self.sigma_ema.cpu().tolist(), 'xmax': self.xmax_ema.cpu().tolist()},
        )
        bias_f = self.bias.detach().cpu() if self.bias is not None else None
        return HNLQExportBundle(Wq.detach().cpu(), bias_f, meta)

    @torch.no_grad()
    def to_inference_linear(self):
        bundle = self.export_quantized()
        lin = nn.Linear(self.in_dim, self.out_dim, bias=(bundle.bias_f is not None))
        lin.weight.copy_(bundle.weight_q.to(lin.weight.dtype))
        if bundle.bias_f is not None:
            lin.bias.copy_(bundle.bias_f.to(lin.bias.dtype))
        return lin

    @torch.no_grad()
    def save_export(self, path_prefix):
        bundle = self.export_quantized()
        torch.save(bundle.weight_q, f"{path_prefix}_weight_q.pt")
        if bundle.bias_f is not None:
            torch.save(bundle.bias_f, f"{path_prefix}_bias_f.pt")
        torch.save(bundle.meta, f"{path_prefix}_meta.pt")