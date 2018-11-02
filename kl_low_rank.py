import torch
import numpy as np

class LowRank:
    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.R = torch.zeros(N,1).float().cuda()
        self.C = torch.zeros(1,D).float().cuda()
        print("Low-Rank", N, D)

    def update(self, gradient, beta1, beta2=1.0):
        self.R = beta1 * self.R + beta2 * torch.mean(gradient, dim=1, keepdim=True)
        self.C = beta1 * self.C + beta2 * torch.mean(gradient, dim=0, keepdim=True)
        result = torch.matmul(self.R, self.C) / torch.mean(self.R)
        return result
