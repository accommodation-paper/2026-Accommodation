import torch


def compatibility_operator(mu_x, sigma_x, mu_p, sigma_p, eps=1e-8, kappa=2.5, delta=1e-4):
    sigma_x = torch.nn.functional.softplus(sigma_x) + eps
    sigma_p = torch.nn.functional.softplus(sigma_p) + eps
    delta_t = torch.tensor(delta, device=mu_x.device, dtype=mu_x.dtype)
    containment = sigma_p - sigma_x - torch.abs(mu_x - mu_p)
    s = torch.sigmoid(containment / kappa)
    denom = sigma_x**2 + sigma_p**2 + eps
    bc = torch.sqrt((2 * sigma_x * sigma_p) / denom) * torch.exp(-0.25 * (mu_x - mu_p)**2 / denom)
    log_score = s * torch.log1p(delta_t) + (1 - s) * torch.log(bc + delta_t)
    return torch.exp(log_score)
