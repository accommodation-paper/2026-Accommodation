import torch
from torch import nn
import torch.nn.functional as F


class AccommodationLoss(nn.Module):
	def __init__(self, differentiation_lambda: float = 0.5):
		super().__init__()
		self.differentiation_lambda = differentiation_lambda
		self.lambda_ce_loss = 1 - differentiation_lambda

	def differentation_loss(self, differentiation_tensor: torch.Tensor):
		target = torch.eye(differentiation_tensor.size(0), device=differentiation_tensor.device).unsqueeze(-1).repeat(1, 1, differentiation_tensor.size(-1))
		return F.mse_loss(differentiation_tensor, target)

	def forward(self, differentiation_tensor: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
		differentiation_loss = self.differentation_loss(differentiation_tensor)
		ce_loss = F.cross_entropy(logits, labels)
		total = self.differentiation_lambda * differentiation_loss + self.lambda_ce_loss * ce_loss
		return total
