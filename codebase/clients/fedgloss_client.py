import copy
import torch
from collections import OrderedDict

from .sam_client import SAMClient
from .minimizers import FedGloSSOpt


class FedGloSSClient(SAMClient):
    def __init__(self, seed, client_id, lr, weight_decay, batch_size, momentum, train_data, eval_data, model, dataset, rho_l, eta, beta, device=None,
                 num_workers=0, run=None):
        super().__init__(seed, client_id, lr, weight_decay, batch_size, momentum, train_data, eval_data, model, dataset, rho_l, eta, device,
                 num_workers, run)
        self.beta = beta
        self.state = OrderedDict()
        
        # initialize local state
        for n, p in self.model.named_parameters():
            self.state[n] = torch.zeros_like(p)

    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        num_train_samples, update = super().train(num_epochs, batch_size, minibatch)
        return num_train_samples, update

    def run_epoch(self, optimizer, criterion):
        minimizer = FedGloSSOpt(optimizer, self.model, self.rho, self.eta, self.beta)
        
        running_loss = 0.0
        i = 0
        
        # save the server model
        initial_model = OrderedDict()
        for n, p in self.model.named_parameters():
            initial_model[n] = copy.deepcopy(torch.clone(p).detach())
        
        # training
        for t, (inputs, targets) in enumerate(self.trainloader):
            inputs = inputs.to(self.device)
            inputs = self.train_transforms[self.dataset](inputs)
            targets = targets.to(self.device)

            # Ascent Step
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            minimizer.ascent_step()

            # Descent Step
            criterion(self.model(inputs), targets).backward()
            minimizer.descent_step(initial_model, self.state)

            with torch.no_grad():
                running_loss += loss.item()

            i += 1
        
        final_model = copy.deepcopy(self.model.state_dict())

        # solve the lagrangian function on the client side
        for n in self.state.keys():
            self.state[n] -= (final_model[n] - initial_model[n]) / self.beta
        
        if i == 0:
            print("Not running epoch", self.id)
            return 0
        return running_loss / i