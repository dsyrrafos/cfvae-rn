import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarFlow(nn.Module):
    """
    z -> z' = z + u * h(w^T z + b)
    where h = tanh
    """

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.u = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def m(self, x):
        # Softplus to enforce w^T u > -1 for invertibility
        return -1 + F.softplus(x)

    def forward(self, z):
        """
        Returns:
            z_k: transformed latent
            log_abs_det_jacobian: log determinant of the flow Jacobian
        """
        # z: (batch, dim)
        wzb = torch.matmul(z, self.w.t()) + self.b  # (batch, 1)
        h = torch.tanh(wzb)                          # (batch, 1)
        
        # Compute psi(z) = h'(w^T z + b) * w
        h_prime = 1 - h.pow(2)                       # derivative of tanh
        psi = h_prime * self.w                       # (batch, dim)

        # Enforce invertibility: u_hat = u + (m(w^T u) - w^T u) * w / ||w||^2
        wtu = torch.matmul(self.w, self.u.t())       # scalar
        u_hat = self.u + (self.m(wtu) - wtu) * self.w / (self.w.norm()**2 + 1e-8)

        # z' = z + u_hat * h
        z_new = z + h * u_hat

        # log|det(dz'/dz)| = log|1 + psi^T u_hat|
        log_det = torch.log(torch.abs(1 + torch.matmul(psi, u_hat.t()) ) + 1e-8)
        log_det = log_det.squeeze(-1)

        return z_new, log_det


class FlowSequential(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        total_logdet = 0.0
        for flow in self.flows:
            z, logdet = flow(z)
            total_logdet += logdet
        return z, total_logdet

