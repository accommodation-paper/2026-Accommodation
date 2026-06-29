import torch
import torch.nn as nn

from accommodation.model.compatibility_operator import compatibility_operator


class ImageModelPerturbator:
    def __init__(self, model: nn.Module):
        self.model = model

    def forward(self, x, perturbation_factor, perturbation_config: dict = None):
        h = self.model.backbone(x)
        x_mu = self.model.accommodation_layer.mu_encoder(h).unsqueeze(1).unsqueeze(2)
        x_sigma = self.model.accommodation_layer.sigma_encoder(h).unsqueeze(1).unsqueeze(2)
        c_mu = self._processed_positive_mu(perturbation_factor, perturbation_config)
        c_sigma = self.model.accommodation_layer.positive_sigma.unsqueeze(0)
        similitude = compatibility_operator(x_mu, x_sigma, c_mu, c_sigma)
        fit = self.model.accommodation_layer.policy(similitude.clamp(min=0.0, max=1.0) + 1e-12, self.model.accommodation_layer, is_neg=False)
        if self.model.accommodation_layer.negative_potents:
            c_negative_mu = self._processed_negative_mu(perturbation_factor, perturbation_config)
            c_negative_sigma = self.model.accommodation_layer.negative_sigma.unsqueeze(0)
            negative_similitude = compatibility_operator(x_mu, x_sigma, c_negative_mu, c_negative_sigma)
            negative_fit = self.model.accommodation_layer.policy(negative_similitude.clamp(min=0.0, max=1.0) + 1e-12, self.model.accommodation_layer, is_neg=True)
            fit = fit - negative_fit
        return self.model.accommodation_layer.pool(fit), self.model.accommodation_layer.calc_differentiation_tensor()

    def _processed_positive_mu(self, perturbation_factor, perturbation_config):
        positive_perturb_config = perturbation_config['positive']
        if positive_perturb_config['perturb'] and not positive_perturb_config['mask']:
            return (
                self.model.accommodation_layer.positive_mu.unsqueeze(0)
                + perturbation_factor * torch.randn_like(self.model.accommodation_layer.positive_mu.unsqueeze(0))
            )

        elif positive_perturb_config['perturb']:
            mask = positive_perturb_config['mask'].unsqueeze(0).unsqueeze(-1)
            return (
                self.model.accommodation_layer.positive_mu.unsqueeze(0)
                + perturbation_factor * mask * torch.randn_like(self.model.accommodation_layer.positive_mu.unsqueeze(0))
            )
        return self.model.accommodation_layer.positive_mu.unsqueeze(0)

    def _processed_negative_mu(self, perturbation_factor, perturbation_config):
        negative_perturb_config = perturbation_config['negative']
        if negative_perturb_config['perturb'] and not negative_perturb_config['mask']:
            return (
                self.model.accommodation_layer.negative_mu.unsqueeze(0)
                + perturbation_factor * torch.randn_like(self.model.accommodation_layer.negative_mu.unsqueeze(0))
            )

        elif negative_perturb_config['perturb']:
            mask = negative_perturb_config['mask'].unsqueeze(0).unsqueeze(-1)
            return (
                self.model.accommodation_layer.negative_mu.unsqueeze(0)
                + perturbation_factor * mask * torch.randn_like(self.model.accommodation_layer.negative_mu.unsqueeze(0))
            )
        return self.model.accommodation_layer.negative_mu.unsqueeze(0)


class TextModelPerturbator:
    def __init__(self, model: nn.Module):
        self.model = model

    def forward(self, x, att_mask, perturbation_factor, perturbation_config: dict = None):
        h = self.model.encode(x, att_mask)
        x_mu = self.model.accommodation_layer.mu_encoder(h).unsqueeze(1).unsqueeze(2)
        x_sigma = self.model.accommodation_layer.sigma_encoder(h).unsqueeze(1).unsqueeze(2)
        c_mu = self._processed_positive_mu(perturbation_factor, perturbation_config)
        c_sigma = self.model.accommodation_layer.positive_sigma.unsqueeze(0)
        similitude = compatibility_operator(x_mu, x_sigma, c_mu, c_sigma)
        fit = self.model.accommodation_layer.policy(similitude.clamp(min=0.0, max=1.0) + 1e-12, self.model.accommodation_layer, is_neg=False)
        if self.model.accommodation_layer.negative_potents:
            c_negative_mu = self._processed_negative_mu(perturbation_factor, perturbation_config)
            c_negative_sigma = self.model.accommodation_layer.negative_sigma.unsqueeze(0)
            negative_similitude = compatibility_operator(x_mu, x_sigma, c_negative_mu, c_negative_sigma)
            negative_fit = self.model.accommodation_layer.policy(negative_similitude.clamp(min=0.0, max=1.0) + 1e-12, self.model.accommodation_layer, is_neg=True)
            fit = fit - negative_fit
        return self.model.accommodation_layer.pool(fit), self.model.accommodation_layer.calc_differentiation_tensor()

    def _processed_positive_mu(self, perturbation_factor, perturbation_config):
        positive_perturb_config = perturbation_config['positive']
        if positive_perturb_config['perturb'] and not positive_perturb_config['mask']:
            return (
                self.model.accommodation_layer.positive_mu.unsqueeze(0)
                + perturbation_factor * torch.randn_like(self.model.accommodation_layer.positive_mu.unsqueeze(0))
            )

        elif positive_perturb_config['perturb']:
            mask = positive_perturb_config['mask'].unsqueeze(0).unsqueeze(-1)
            return (
                self.model.accommodation_layer.positive_mu.unsqueeze(0)
                + perturbation_factor * mask * torch.randn_like(self.model.accommodation_layer.positive_mu.unsqueeze(0))
            )
        return self.model.accommodation_layer.positive_mu.unsqueeze(0)

    def _processed_negative_mu(self, perturbation_factor, perturbation_config):
        negative_perturb_config = perturbation_config['negative']
        if negative_perturb_config['perturb'] and not negative_perturb_config['mask']:
            return (
                self.model.accommodation_layer.negative_mu.unsqueeze(0)
                + perturbation_factor * torch.randn_like(self.model.accommodation_layer.negative_mu.unsqueeze(0))
            )

        elif negative_perturb_config['perturb']:
            mask = negative_perturb_config['mask'].unsqueeze(0).unsqueeze(-1)
            return (
                self.model.accommodation_layer.negative_mu.unsqueeze(0)
                + perturbation_factor * mask * torch.randn_like(self.model.accommodation_layer.negative_mu.unsqueeze(0))
            )
        return self.model.accommodation_layer.negative_mu.unsqueeze(0)