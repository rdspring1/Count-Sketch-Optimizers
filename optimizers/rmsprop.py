import math
import operator
from functools import reduce

import torch
from torch.optim import Optimizer

from dense_exp_cms import DenseCMS
from exp_cms_flat import CountMinSketch

class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.999, eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter : {}".format(beta))
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)

    def dense(self, p, grad, group):
        state = self.state[p]

        # State initialization
        if len(state) == 0:
           state['step'] = 0
           # Exponential moving average of squared gradient values
           size = p.data.size()
           N = size[0]
           D = max(reduce(operator.mul, [size[i] for i in range(1,len(size),1)], 1), 1)
           state['exp_avg_sq'] = DenseCMS(N, D) 

        exp_avg_sq = state['exp_avg_sq']
        beta = group['beta']
        state['step'] += 1

        if group['weight_decay'] != 0:
           grad = grad.add(group['weight_decay'], p.data)

        if state['step'] % 1000 == 0:
           #print("Cleaning")
           exp_avg_sq.clean(0.25)

        bias_correction = 1 - beta ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction)
        exp_avg_sq.update(p, grad, -step_size, beta)

    def sparse(self, p, grad, group):
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of squared gradient values
            N, D = grad.data.size()
            state['exp_avg_sq'] = CountMinSketch(N, D)

        exp_avg_sq = state['exp_avg_sq']
        beta = group['beta']
        state['step'] += 1

        if group['weight_decay'] != 0:
           grad = grad.add(group['weight_decay'], p.data)

        grad = grad.coalesce()  # the update is non-linear so indices must be unique
        grad_indices = grad._indices()
        grad_values = grad._values()
        size = grad.size()

        def make_sparse(values):
            constructor = grad.new
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, size)

        if state['step'] % 1000 == 0:
           #print("Cleaning")
           exp_avg_sq.clean(0.25)

        # Decay the first and second moment running average coefficient
        #      old <- b * old + (1 - b) * new  <==> old += (1 - b) * (new - old)
        exp_avg_sq_update = exp_avg_sq.update(grad_indices, grad_values.pow(2), size, beta)._values()
        denom = exp_avg_sq_update.sqrt_().add_(group['eps'])
        update = grad.values() / denom

        bias_correction = 1 - beta ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction)
        p.data.add_(make_sparse(-step_size * update))

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    self.sparse(p, grad, group)
                else:
                    self.dense(p, grad, group)
        return loss
