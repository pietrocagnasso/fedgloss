import torch
from collections import defaultdict


class SAM:
    def __init__(self, optimizer, model, rho, eta):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
    
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class FedGloSSOpt(SAM):
    def __init__(self, optimizer, model, rho, eta, beta):
        super().__init__(optimizer, model, rho, eta)
        self.beta = beta

    @torch.no_grad()
    def descent_step(self, initial_model, local_state):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            
            # remove the perturbation
            p.sub_(self.state[p]["eps"])
            
            # add components to solve the lagrangian function
            p.grad.sub_(local_state[n])
            p.grad.add_((p - initial_model[n]) / self.beta)
        self.optimizer.step()
        self.optimizer.zero_grad()
