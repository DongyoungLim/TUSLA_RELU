import torch
import math
from torch.optim.optimizer import Optimizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TUSLA(Optimizer):

    def __init__(self, params, lr=1e-1, eta=1e-4, beta=1e10, r=3):

        defaults = dict(lr=lr, beta=beta, eta=eta, r=r)
        super(TUSLA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TUSLA, self).__setstate__(state)


    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            pnorm = 0
            for p in group['params']:
                #pnorm += torch.sum(torch.pow(p.data, exponent=2))
                pnorm += torch.norm(p)
            r = group['r']
            total_norm = torch.pow(pnorm, 2*r)


            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0

                eta, beta, lr = group['eta'], group['beta'], group['lr']

                state['step'] += 1

                noise = math.sqrt(2 * lr / beta) * torch.randn(size=p.size(), device=device)

                numer = grad + eta * p * total_norm
                denom = 1 + math.sqrt(lr) * total_norm

                p.data.addcdiv_(value=-lr, tensor1=numer, tensor2=denom).add_(noise)

        return loss

