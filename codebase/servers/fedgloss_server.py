import copy
import torch
from collections import OrderedDict
import warnings

from .fedopt_server import FedOptServer

class FedGloSSServer(FedOptServer):
    def __init__(self, client_model, server_opt, server_lr, rho, rho_l, rho0, T_s, T, beta, tot_clients,
                 momentum=0, opt_ckpt=None):
        super().__init__(client_model, server_opt, server_lr, momentum, opt_ckpt)
        
        self.beta = beta
        self.tot_clients = tot_clients
        self.delta_w_tilde = None
        self.epsilon_tilde = None
        self.t = 0

        # initialize the server state to 0
        self.sigma = OrderedDict()
        for n, p in client_model.named_parameters():
            self.sigma[n] = torch.zeros_like(p)
        
        # SCHEDULING: setup rho values for both client and server
        if T_s > 1:
            if T_s > T:
                warnings.warn("### The value that was set for T_s is higher tha the number of rounds T ###")

            # define clients' rho values across rounds
            self.rhos_l = [rho0]
            delta = (rho_l - rho0) / T_s
            for _ in range(T_s):
                self.rhos_l.append(self.rhos_l[-1] + delta)
            self.rhos_l += [rho_l] * (T - len(self.rhos_l))

            # define server's rho values across rounds
            self.server_rhos = [rho0]
            delta = (rho - rho0) / T_s
            for _ in range(T_s):
                self.server_rhos.append(self.rhos_l[-1] + delta)
            self.server_rhos += [rho] * (T - len(self.server_rhos))
        else:
            self.rhos_l = [rho_l] * T
            self.server_rhos = [rho] * T

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, analysis=False):
        # perform the ascent step excluding the first round
        if self.delta_w_tilde is not None:
            self.ascent_step()
        
        if clients is None:
            clients = self.selected_clients
        
        # rho scheduling
        for c in clients:
            c.rho = self.rhos_l[self.t]
        
        sys_metrics = super().train_model(num_epochs, batch_size, minibatch, clients, analysis)
        return sys_metrics

    def update_model(self): 
        # update the server state sigma
        for n in self.sigma:
            deviation = torch.zeros_like(self.sigma[n])
            for update in self.updates:
                deviation += update[1][n]
            
            self.sigma[n] -= deviation / (self.beta * self.tot_clients * len(self.updates))

        # perfor the descent step
        if self.epsilon_tilde is not None:
            self.remove_perturbation()
        
        # update the global model
        self.client_model.load_state_dict(self.model)
        pseudo_gradient = self._average_updates()
        self._update_global_model_gradient(pseudo_gradient)
        self.model = copy.deepcopy(self.client_model.state_dict())
        
        # save the pseuodo gradient to compute next step's perturbation
        self.save_pseudograd()
        
        # solve the lagrangian function on the server side
        for n in self.sigma:
            self.model[n] -= self.beta * self.sigma[n]
        
        # METRICS
        # compute the client drift metric
        norms = []
        for _, update in self.updates:
            norms.append(0.0)
            for k in update:
                norms[-1] += torch.norm(update[k], p=2) ** 2
            norms[-1] = (norms[-1] ** 0.5).cpu()
        # compute pseudo-gradient's norm
        self.total_grad = self._get_model_total_grad()

        self.updates = []
        self.t += 1
        return
    
    def ascent_step(self):
        self.epsilon_tilde = OrderedDict()
        
        # compute delta_w's norm
        norm = []
        for k in self.delta_w_tilde:
            norm.append(torch.norm(self.delta_w_tilde[k], p=2))
        norm = torch.norm(torch.stack(norm), p=2)

        # compute component-by-component epsilon_tilde
        for k in self.model:
            e = torch.clone(self.delta_w_tilde[k].mul(self.server_rhos[self.t]/norm))
            self.epsilon_tilde[k] = torch.clone(e)

            # apply the perturbation
            self.model[k].add_(e)            

    def remove_perturbation(self):
        for k in self.model:
            self.model[k].sub_(self.epsilon_tilde[k])
    
    def save_pseudograd(self):
        if self.delta_w_tilde is None:
            self.delta_w_tilde = OrderedDict()
        for n, p in self.client_model.named_parameters():
            self.delta_w_tilde[n] = torch.clone(p.grad).detach()
