import torch
import numpy as np

class LowRank:
    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.R = None
        self.C = None
        print("L2 Low-Rank", N, D)

    def update(self, update, beta1, beta2=1.0):
        if self.R is None and self.C is None:
            matrix = torch.zeros(N,D).float().cuda()
        else:
            matrix = torch.matmul(torch.matmul(R, S), C)
        matrix.mul_(beta1).add_(beta2, update)
        u, s, v = torch.svd(matrix)
        self.R = u[:,:1]
        self.C = v[:,:1].t()
        self.S = torch.diag(s)[:1,:1]
        return matrix
