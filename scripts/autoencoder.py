import torch
import torch.nn as nn
import torch.nn.functional as F
from routenet import RouteNetEncoder, RouteNetPriorEncoder
from helpers import kl_divergence_gaussians
from flows import PlanarFlow, FlowSequential


class LearntPriorVariationalRouteNet(nn.Module):
    """
    Variational Autoencoder with learnt prior over path latent space using RouteNet.
    Args:
        input_dim_path: Dimension of path input features
        input_dim_link: Dimension of link input features
        output_dim: Dimension of output (e.g., delay prediction)
        iterations: Number of message-passing iterations in RouteNet
        link_state_dim: Dimension of link state in RouteNet
        path_state_dim: Dimension of path state in RouteNet
        aggr_dropout: Dropout rate for aggregation layers
        readout_dropout: Dropout rate for readout layers
        latent_dim: Dimension of the latent space
        conditional_encoder: Optional RouteNetEncoder for conditional VAE
        device: Device to run the model on ('cpu' or 'cuda')
    """
    
    def __init__(self, input_dim_path, input_dim_link, output_dim, iterations, link_state_dim, path_state_dim, aggr_dropout, readout_dropout, latent_dim, conditional_encoder = None, device='cpu'):
        super(LearntPriorVariationalRouteNet, self).__init__()
        self.input_dim_path = input_dim_path
        self.input_dim_link = input_dim_link
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.device = device

        # Posterior network to learn posterior distribution over path latent space
        self.encoder = RouteNetEncoder(input_dim_path+output_dim, input_dim_link, output_dim, iterations, link_state_dim, path_state_dim, aggr_dropout, readout_dropout, device)
        self.fc_posterior_mu = nn.Linear(path_state_dim, latent_dim)
        self.fc_posterior_logvar = nn.Linear(path_state_dim, latent_dim)
        self.conditional_encoder = conditional_encoder
        self.condition_dim = 0 if conditional_encoder is None else path_state_dim

        # Prior network to learn prior distribution over path latent space
        self.prior_network = RouteNetPriorEncoder(input_dim_path, input_dim_link, output_dim, iterations, link_state_dim, path_state_dim, aggr_dropout, readout_dropout, device)
        self.fc_prior_mu = nn.Linear(path_state_dim, latent_dim)
        self.fc_prior_logvar = nn.Linear(path_state_dim, latent_dim)

        # Readout network to predict delay from path latent representation
        self.readout = nn.Sequential(
            nn.Linear(self.condition_dim + latent_dim, path_state_dim // 2),
            nn.ReLU(),
            nn.Dropout(readout_dropout),
            nn.Linear(path_state_dim // 2, path_state_dim // 4),
            nn.ReLU(),
            nn.Dropout(readout_dropout),
            nn.Linear(path_state_dim // 4, self.output_dim)
        )

    def encode(self, data):
        x_g_posterior, _ = self.encoder(data)
        mu_posterior = self.fc_posterior_mu(x_g_posterior)
        logvar_posterior = self.fc_posterior_logvar(x_g_posterior)
        return mu_posterior, logvar_posterior

    def encode_prior(self, inputs):
        x_g_prior, _ = self.prior_network(inputs)
        mu_prior = self.fc_prior_mu(x_g_prior)
        logvar_prior = self.fc_prior_logvar(x_g_prior)
        return mu_prior, logvar_prior
    
    def decode(self, z, c):
        if c is not None:
            z = torch.cat((z, c.squeeze(0)), dim=1)  # Concatenate path latent and conditional code
        prediction = self.readout(z).squeeze()  # Predict delay from path latent representation
        return prediction

    def reparameterize(self, mu, logvar, eps_scale=1.):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std) * eps_scale
        return eps.mul(std).add_(mu)
    
    def encode_condition(self, inputs):
        with torch.no_grad():
            context = self.conditional_encoder(inputs)
        return context
    
    # Forward pass for inference
    def forward(self, data, return_latent=False, use_posterior=False):
        inputs, _ = data
        if self.conditional_encoder is not None:
            _, conditional_code = self.encode_condition(inputs)
        else:
            conditional_code = None
        
        if use_posterior:
            mu, logvar = self.encode(data)
        else:
            mu, logvar = self.encode_prior(inputs)

        z = self.reparameterize(mu, logvar)

        prediction = self.decode(z, conditional_code).float()

        if return_latent:
            return prediction, (z, conditional_code)

        return prediction

    # Loss function for training
    def loss_function(self, data, targets, beta=0.05, reduction='mean'):
        inputs, labels = data
        if self.conditional_encoder is not None:
            _, conditional_code = self.encode_condition(inputs)
        else:
            conditional_code = None

        mu_posterior, logvar_posterior = self.encode(data)
        z_posterior = self.reparameterize(mu_posterior, logvar_posterior)

        mu_prior, logvar_prior = self.encode_prior(inputs)

        if self.conditional_encoder is not None:
            prediction = self.decode(z_posterior, conditional_code).float()
        else:
            prediction = self.decode(z_posterior, None).float()

        ground_truth = torch.stack([labels[t] for t in targets], dim=2).squeeze().float().to(self.device)
        
        pred_loss = F.mse_loss(prediction, ground_truth, reduction=reduction)

        kld = kl_divergence_gaussians(mu_posterior, logvar_posterior, mu_prior, logvar_prior)

        loss = pred_loss + beta*kld

        return loss, pred_loss, kld
    
    # Loss function for evaluation
    def pred_loss_function(self, data, targets, reduction='mean'):
        inputs, labels = data
        if self.conditional_encoder is not None:
            _, conditional_code = self.encode_condition(inputs)
        else:
            conditional_code = None

        mu_prior, logvar_prior = self.encode_prior(inputs)
        x_g_prior = self.reparameterize(mu_prior, logvar_prior)

        if self.conditional_encoder is not None:
            prediction = self.decode(x_g_prior, conditional_code).float()
        else:
            prediction = self.decode(x_g_prior, None).float()

        ground_truth = torch.stack([labels[t] for t in targets], dim=2).squeeze().float().to(self.device)
        
        pred_loss = F.mse_loss(prediction, ground_truth, reduction=reduction)

        return pred_loss

# Variational Autoencoder with Normalizing Flows (Inherit from LearntPriorVariationalRouteNet)
class FlowLearntPriorVariationalRouteNet(LearntPriorVariationalRouteNet):
    """
    VAE + Planar Flows on the posterior.
    Inherits entire architecture from original class.
    """

    def __init__(self, *args, latent_dim, flow_type='planar', n_flows=4, **kwargs):
        super().__init__(*args, latent_dim=latent_dim, **kwargs)
        self.flow_type = flow_type
        self.latent_dim = latent_dim
        self.n_flows = n_flows

        if flow_type == 'planar':
            # Create planar flows
            flows = [PlanarFlow(self.latent_dim) for _ in range(n_flows)]
            self.flow = FlowSequential(flows)
        else:
            raise ValueError(f"Unsupported flow type: {flow_type}")

    def flow_transform(self, z):
        """Apply the sequence of flows to z."""
        return self.flow(z)

    def forward(self, inputs, return_latent=False):
        # Same as before: inference uses the *prior*, no flows
        return super().forward(inputs, return_latent)

    def loss_function(self, data, targets, beta=0.05, reduction='mean'):
        """
        Now includes flow-transformed posterior + log-det-Jacobian term.
        KL is still Gaussian-to-Gaussian (flows come on top).
        """
        inputs, labels = data
        if self.conditional_encoder is not None:
            _, conditional_code = self.encode_condition(inputs)
        else:
            conditional_code = None

        # 1) Encode posterior
        mu_posterior, logvar_posterior = self.encode(data)
        z0 = self.reparameterize(mu_posterior, logvar_posterior)

        # 2) Apply flows: zK, sum(log|det dflow|)
        zk, logdet = self.flow_transform(z0)

        # 3) Encode prior
        mu_prior, logvar_prior = self.encode_prior(inputs)

        # 4) Decode using flow-transformed latent
        if self.conditional_encoder is not None:
            prediction = self.decode(zk, conditional_code).float()
        else:
            prediction = self.decode(zk, None).float()

        # 5) Reconstruction
        gt = torch.stack([labels[t] for t in targets], dim=2).squeeze().float().to(self.device)
        pred_loss = F.mse_loss(prediction, gt, reduction=reduction)

        # 6) Base KL (Gaussian-to-Gaussian)
        kl_base = kl_divergence_gaussians(mu_posterior, logvar_posterior,
                                          mu_prior, logvar_prior)

        # 7) Flow-corrected KL
        #    KL(qK(z) || p(z)) = KL_base - E[logdet]
        kld = kl_base - logdet.mean()

        # 8) Final loss
        loss = pred_loss + beta * kld

        return loss, pred_loss, kld

