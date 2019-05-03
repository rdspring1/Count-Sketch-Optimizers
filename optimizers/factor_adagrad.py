import torch
from torch.optim import Optimizer

class LowRank:
    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.v_r = torch.zeros(N,1).float().cuda()
        self.v_c = torch.zeros(1,D).float().cuda()
        self.step_num = 0
        print("Factorized Adagrad Low Rank", N, D)

    def update(self, value):
        self.v_r = self.v_r + torch.mean(value, dim=1, keepdim=True)
        self.v_c = self.v_c + torch.mean(value, dim=0, keepdim=True)
        return torch.matmul(self.v_r, self.v_c) / torch.mean(self.v_r)

class Adagrad(Optimizer):
    """Implements Adagrad algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))

        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(Adagrad, self).__init__(params, defaults)

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
                state = self.state[p]

                if 'sum' not in state and 'step' not in state:
                    state['step'] = 0
                    if grad.is_sparse:
                        N, D = grad.data.size()
                        state['sum'] = LowRank(N, D)
                    else:
                        state['sum'] = torch.full_like(grad.data, group['initial_accumulator_value'])


                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    grad_dense = grad.to_dense()
                    std = state['sum'].update(grad_dense.pow(2)).sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    update = grad_values / std_values
                    p.data.add_(make_sparse(-clr * update))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)
        return loss
