import torch
from torch import nn
from enum import Enum
from accommodation.model.policies.bayes import bayes_policy
from accommodation.model.policies.likelihood import likelihood_policy
from accommodation.model.compatibility_operator import compatibility_operator

class Policy(Enum):
	Bayes = 1
	Likelihood = 2


class Pool(Enum):
	Mean = 1
	Max = 2
	MeanMax = 3
	MaxMean = 4
	LSE = 5


class AccommodationLayer(nn.Module):
	def __init__(self,
				 in_features: int,
				 num_classes: int,
				 latent_dim: int,
				 num_potents_per_class: int = 1,
				 negative_potents: bool = True,
				 neutral_potents: int = 1,
				 plasticity: bool = True,
				 policy: Policy = Policy.Likelihood,
				 pool: Pool = Pool.MaxMean,
				 ):
		super().__init__()
		self._in_features = in_features
		self.num_classes = num_classes
		self._latent_dim = latent_dim
		self._num_potents_per_class = num_potents_per_class
		self.negative_potents = negative_potents
		self._neutral_potents = neutral_potents
		self.plasticity = plasticity
		self._set_policy(policy)
		self._set_pool(pool)
		self._set_potents()
		self.mu_encoder = nn.Linear(in_features, latent_dim)
		self.sigma_encoder = nn.Linear(in_features, latent_dim)
		if self.plasticity:
			self._init_plasticity()
			self._init_specialization()
			self._register_plasticity_hooks()

	def _set_policy(self, policy):
		if policy == Policy.Bayes:
			self.policy = bayes_policy
		elif policy == Policy.Likelihood:
			self.policy = likelihood_policy
		else:
			raise Exception("The policy must be equal to bayes or likelihood")

	def _set_pool(self, pool):
		if pool == Pool.Mean:
			self.pool = lambda x: x.mean(dim=(-1, -2))
		elif pool == Pool.Max:
			self.pool = lambda x: x.amax(dim=(-1, -2))
		elif pool == Pool.MeanMax:
			self.pool = lambda x: x.mean(dim=(-1)).amax(dim=(-1))
		elif pool == Pool.MaxMean:
			self.pool = lambda x: x.amax(dim=(-1)).mean(dim=(-1))
		elif pool == Pool.LSE:
			self.pool = lambda x: torch.logsumexp(x, dim=(-1, -2))
		else:
			raise Exception("The pool method must be equal to mean, max, mean-max, max-mean or lse")

	def _set_potents(self):
		if self._num_potents_per_class < 1: raise Exception("The number of potents must be at least 1")
		self._set_positive_potents()
		self._set_negative_potents()
		self._set_neutral_potents()

	def _set_positive_potents(self):
		self.positive_mu = nn.Parameter(
			torch.randn(self.num_classes, self._num_potents_per_class, self._latent_dim)
		)
		self.positive_sigma = nn.Parameter(
			torch.ones(self.num_classes, self._num_potents_per_class, self._latent_dim)
		)

	def _set_negative_potents(self):
		if self.negative_potents:
			self.negative_mu = nn.Parameter(
				torch.randn(self.num_classes, self._num_potents_per_class, self._latent_dim)
			)
			self.negative_sigma = nn.Parameter(
				torch.ones(self.num_classes, self._num_potents_per_class, self._latent_dim)
			)
		else:
			self.register_parameter("negative_mu", None)
			self.register_parameter("negative_sigma", None)

	def _set_neutral_potents(self):
		if self._neutral_potents > 0:
			self.neutral_mu = nn.Parameter(
				torch.randn(self._neutral_potents, self._num_potents_per_class, self._latent_dim)
			)
			self.neutral_sigma = nn.Parameter(
				torch.ones(self._neutral_potents, self._num_potents_per_class, self._latent_dim)
			)
		else:
			self.register_parameter("neutral_mu", None)
			self.register_parameter("neutral_sigma", None)

	def _init_plasticity(self):
		self.register_buffer(
			"positive_plasticity",
			torch.ones(self.num_classes, self._num_potents_per_class)
		)
		self.register_buffer(
			"positive_usage",
			torch.zeros(self.num_classes, self._num_potents_per_class)
		)
		if self.negative_potents:
			self.register_buffer(
				"negative_plasticity",
				torch.ones(self.num_classes, self._num_potents_per_class)
			)
			self.register_buffer(
				"negative_usage",
				torch.zeros(self.num_classes, self._num_potents_per_class)
			)

	def _init_specialization(self):
		self.register_buffer(
			"positive_specialization",
			torch.zeros(self.num_classes, self._num_potents_per_class)
		)

		if self.negative_potents:
			self.register_buffer(
				"negative_specialization",
				torch.zeros(self.num_classes, self._num_potents_per_class)
			)

	def _scale_neg_grad(self, grad):
		return grad * self.negative_plasticity.unsqueeze(-1).to(grad.dtype)

	def scale_pos_grad(self, grad):
		return grad * self.positive_plasticity.unsqueeze(-1).to(grad.dtype)

	def _register_plasticity_hooks(self):
		self.positive_mu.register_hook(self.scale_pos_grad)
		self.positive_sigma.register_hook(self.scale_pos_grad)
		if self.negative_potents:
			self.negative_mu.register_hook(self._scale_neg_grad)
			self.negative_sigma.register_hook(self._scale_neg_grad)

	@torch.no_grad()
	def reset_plasticity_stats(self):
		if not self.plasticity:
			return
		self.positive_usage.zero_()
		if self.negative_potents:
			self.negative_usage.zero_()

	@torch.no_grad()
	def update_plasticity(
			self,
			gamma: float = 2.0,
			p_min: float = 0.02,
			p_max: float = 1.0,
	):
		if not self.plasticity:
			return
		pos_norm = self.positive_usage.sum(dim=1, keepdim=True).clamp_min(1.0)
		pos_freq = self.positive_usage / pos_norm
		target_pos = (1.0 - pos_freq).pow(gamma).clamp(p_min, p_max)
		self.positive_plasticity.copy_(torch.minimum(self.positive_plasticity, target_pos))
		if self.negative_potents:
			neg_norm = self.negative_usage.sum(dim=1, keepdim=True).clamp_min(1.0)
			neg_freq = self.negative_usage / neg_norm
			target_neg = (1.0 - neg_freq).pow(gamma).clamp(p_min, p_max)
			self.negative_plasticity.copy_(torch.minimum(self.negative_plasticity, target_neg))

	def forward(self, x):
		x_mu = self.mu_encoder(x).unsqueeze(1).unsqueeze(2)
		x_sigma = self.sigma_encoder(x).unsqueeze(1).unsqueeze(2)
		c_mu = self.positive_mu.unsqueeze(0)
		c_sigma = self.positive_sigma.unsqueeze(0)
		positive_compatibility_profile = compatibility_operator(x_mu, x_sigma, c_mu, c_sigma)
		evidence_scores = self.policy(positive_compatibility_profile.clamp(min=0.0, max=1.0) + 1e-12, self, is_neg=False)
		if self.negative_potents:
			c_negative_mu = self.negative_mu.unsqueeze(0)
			c_negative_sigma = self.negative_sigma.unsqueeze(0)
			negative_compatibility_profile = compatibility_operator(x_mu, x_sigma, c_negative_mu, c_negative_sigma)
			negative_evidence_scores = self.policy(negative_compatibility_profile.clamp(min=0.0, max=1.0) + 1e-12, self, is_neg=True)
			evidence_scores = evidence_scores - negative_evidence_scores
		return self.pool(evidence_scores), self.calc_differentiation_tensor()

	def calc_differentiation_tensor(self):
		mu_list = [t for t in [self.positive_mu, self.negative_mu, self.neutral_mu] if t is not None]
		sigma_list = [t for t in [self.positive_sigma, self.negative_sigma, self.neutral_sigma] if t is not None]
		mu_all = torch.cat(mu_list, dim=0)
		sigma_all = torch.cat(sigma_list, dim=0)
		mu_flat = mu_all.reshape(-1, self._latent_dim)
		sigma_flat = sigma_all.reshape(-1, self._latent_dim)
		return compatibility_operator(
			mu_flat.unsqueeze(1),
			sigma_flat.unsqueeze(1),
			mu_flat.unsqueeze(0),
			sigma_flat.unsqueeze(0)
		)

	@torch.no_grad()
	def add_info(self, x, y):
		if not self.plasticity: return
		B = x.size(0)
		device = x.device
		b_idx = torch.arange(B, device=device)

		x_mu = self.mu_encoder(x).unsqueeze(1).unsqueeze(2)
		x_sigma = self.sigma_encoder(x).unsqueeze(1).unsqueeze(2)
		sim = compatibility_operator(x_mu, x_sigma,
									 self.positive_mu.unsqueeze(0),
									 self.positive_sigma.unsqueeze(0))
		pos_fit = self.policy(sim.clamp(0.0, 1.0) + 1e-12, self, is_neg=False)
		pos_scores = pos_fit.max(dim=-1).values
		p_pos = pos_scores[b_idx, y].argmax(dim=-1)
		self.positive_usage.index_put_((y, p_pos),
								  torch.ones_like(p_pos, dtype=self.positive_usage.dtype),
								  accumulate=True)

		neg_scores = None
		if self.negative_potents:
			nsim = compatibility_operator(x_mu, x_sigma,
										  self.negative_mu.unsqueeze(0),
										  self.negative_sigma.unsqueeze(0))
			neg_fit = self.policy(nsim.clamp(0.0, 1.0) + 1e-12, self, is_neg=True)
			neg_scores = neg_fit.max(dim=-1).values
			p_neg = neg_scores[b_idx, y].argmax(dim=-1)
			self.negative_usage.index_put_((y, p_neg),
									 torch.ones_like(p_neg, dtype=self.negative_usage.dtype),
									 accumulate=True)

		self._update_specialization(pos_scores, y, neg_scores)

	@torch.no_grad()
	def _update_specialization(self, pos_scores, y, neg_scores=None):
		B, C, K = pos_scores.shape
		mask_in = torch.nn.functional.one_hot(y, num_classes=C).to(pos_scores.dtype)
		mask_out = 1.0 - mask_in
		mask_in = mask_in.unsqueeze(-1)
		mask_out = mask_out.unsqueeze(-1)
		in_sum = (pos_scores * mask_in).sum(dim=0)
		out_sum = (pos_scores * mask_out).sum(dim=0)
		in_cnt = mask_in.sum(dim=0).clamp_min(1.0)
		out_cnt = mask_out.sum(dim=0).clamp_min(1.0)
		in_avg = in_sum / in_cnt
		out_avg = out_sum / out_cnt
		self.positive_specialization.copy_(in_avg - out_avg)
		if neg_scores is not None:
			in_sum = (neg_scores * mask_in).sum(dim=0)
			out_sum = (neg_scores * mask_out).sum(dim=0)
			in_avg = in_sum / in_cnt
			out_avg = out_sum / out_cnt
			self.negative_specialization.copy_(out_avg - in_avg)