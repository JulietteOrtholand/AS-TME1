import torch
import torch.nn as nn


class HighwayModel(nn.Module):
	### Implémentation du module Highway
	### Simplification : C=T-1

	def __init__(self, dim, activ_func=nn.functional.relu, gate_func=nn.functional.softmax):
		super(HighwayModel, self).__init__()
		self.activ_func = activ_func
		self.gate_func = gate_func

		### Mêmes dimensions pour x, H(x, W_H) et T(x W_T)
		self.normal = nn.Linear(dim, dim)
		self.gate = nn.Linear(dim, dim)

	def forward(self, x):
		H = self.activ_func(self.normal(x))
		T = self.gate_func(self.gate(x))
		C = 1-T
		return torch.add( torch.mul(H, T), torch.mul(x, C) )


