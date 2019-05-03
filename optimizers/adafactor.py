import math
import torch
from torch.optim import Optimizer
import numpy as np

class LowRank:
    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.v_r = torch.zeros(N,1).float().cuda()
        self.v_c = torch.zeros(1,D).float().cuda()
        self.step_num = 0
        print("AdaFactor Low-Rank", N, D)

    def decay_rate(self):
        decay_rate = 1. - pow((self.step_num + 1.), -0.8)
        self.step_num += 1
        return decay_rate

    def update(self, value, decay_rate=None):
        if decay_rate is None:
            decay_rate = self.decay_rate()
        self.v_r = decay_rate * self.v_r + (1. - decay_rate) * torch.mean(value, dim=1, keepdim=True)
        self.v_c = decay_rate * self.v_c + (1. - decay_rate) * torch.mean(value, dim=0, keepdim=True)
        return torch.matmul(self.v_r, self.v_c) / torch.mean(self.v_r)

class Adam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)
        #self.exp_avg_sq_error = 0
        #self.count = 1

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def dense(self, p, grad, group):
        amsgrad = group['amsgrad']
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']
        if amsgrad:
           max_exp_avg_sq = state['max_exp_avg_sq']

        state['step'] += 1
        if group['weight_decay'] != 0:
           grad = grad.add(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
        p.data.addcdiv_(-step_size, exp_avg, denom)

    def sparse(self, p, grad, group):
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            N, D = grad.data.size()
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = LowRank(N, D)
            #state['exp_avg_sq_base'] = torch.zeros_like(p.data)

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

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        # Decay the first and second moment running average coefficient
        #      old <- b * old + (1 - b) * new  <==> old += (1 - b) * (new - old)
        old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        exp_avg.add_(make_sparse(exp_avg_update_values))
        numer = exp_avg_update_values.add_(old_exp_avg_values)

        grad_dense = grad.to_dense()
        exp_avg_sq_update_values = exp_avg_sq.update(grad_dense.pow(2).add_(1e-30), beta2).sparse_mask(grad)._values()
        denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
        update = numer / denom
        del exp_avg_update_values, exp_avg_sq_update_values

        '''
        exp_avg_sq_base = state['exp_avg_sq_base']
        old_exp_avg_sq_values_base = exp_avg_sq_base.sparse_mask(grad)._values()
        exp_avg_sq_update_values_base = grad_values.pow(2).sub_(old_exp_avg_sq_values_base).mul_(1 - beta2)
        exp_avg_sq_base.add_(make_sparse(exp_avg_sq_update_values_base))
        exp_avg_sq_values_base = exp_avg_sq_base.sparse_mask(grad)
        exp_avg_sq_approx = exp_avg_sq_update.sparse_mask(grad)

        if (self.count+1) % 200 == 0:
            print(self.exp_avg_sq_error/self.count)
            self.exp_avg_sq_error = 0
            self.count = 0
        self.exp_avg_sq_error += torch.sum(torch.abs(exp_avg_sq_approx-exp_avg_sq_values_base))
        self.count += 1
        '''

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        # Update Clipping
        clipping_denom = max(1.0, Adam.root_mean_square(update))
        update /= clipping_denom

        p.data.add_(make_sparse(-step_size * update))

    def root_mean_square(x):
        return torch.sqrt(torch.mean(x.pow(2))).item()

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
