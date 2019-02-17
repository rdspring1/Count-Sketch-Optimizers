import math
import torch
from torch.optim import Optimizer

from exp_cms import CountMinSketch
from exp_sketch import CountSketch 

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

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def dense(self, p, grad, group):
        amsgrad = group['amsgrad']
        state = self.state[p]
        beta1, beta2 = group['betas']

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            if beta1 > 0:
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        state['step'] += 1
        if group['weight_decay'] != 0:
           grad = grad.add(group['weight_decay'], p.data)

        if beta1 > 0:
            exp_avg = state['exp_avg']
            exp_avg.mul_(beta1).add_(1 - beta1, grad)

        if amsgrad:
           max_exp_avg_sq = state['max_exp_avg_sq']

        # Decay the first and second moment running average coefficient
        exp_avg_sq = state['exp_avg_sq']
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

        if beta1 > 0:
            p.data.addcdiv_(-step_size, exp_avg, denom)
        else:
            p.data.addcdiv_(-step_size, grad, denom)

    def sparse(self, p, grad, group):
        state = self.state[p]
        beta1, beta2 = group['betas']

        # State initialization
        if len(state) == 0:
            N, D = grad.data.size()
            state['step'] = 0
            if beta1 > 0:
                # Exponential moving average of gradient values
                state['exp_avg'] = CountSketch(N, D)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = CountMinSketch(N, D)

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

        if beta1 > 0:
            exp_avg = state['exp_avg']
            numer = exp_avg.update(grad_indices, grad_values, size, beta1)._values()
        else:
            numer = grad._values()

        exp_avg_sq = state['exp_avg_sq']
        if state['step'] % 1000 == 0:
           #print("Cleaning")
           exp_avg_sq.clean(0.25)

        # Decay the first and second moment running average coefficient
        #      old <- b * old + (1 - b) * new  <==> old += (1 - b) * (new - old)
        exp_avg_sq_update = exp_avg_sq.update(grad_indices, grad_values.pow(2), size, beta2)._values()
        denom = exp_avg_sq_update.sqrt_().add_(group['eps'])
        update = numer / denom

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

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
