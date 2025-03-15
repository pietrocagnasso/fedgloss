import torch

from .client import Client
from .minimizers import SAM


class SAMClient(Client):
    def __init__(self, seed, client_id, lr, weight_decay, batch_size, momentum, train_data, eval_data, model, dataset, rho, eta, device=None,
                 num_workers=0, run=None):
        super().__init__(seed, client_id, lr, weight_decay, batch_size, momentum, train_data, eval_data, model, dataset, device,
                         num_workers, run)
        self.rho = rho
        self.eta = eta

    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        num_train_samples, update = super().train(num_epochs, batch_size, minibatch)
        return num_train_samples, update

    def run_epoch(self, optimizer, criterion):
        minimizer = SAM(optimizer, self.model, self.rho, self.eta)
        
        running_loss = 0.0
        i = 0
        for inputs, targets in self.trainloader:
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
            minimizer.descent_step()

            with torch.no_grad():
                running_loss += loss.item()

            i += 1
        if i == 0:
            print("Not running epoch", self.id)
            return 0
        return running_loss / i
