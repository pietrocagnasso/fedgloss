import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import warnings

from baseline_constants import ACCURACY_KEY


class Client:
    train_transforms = {
        "cifar10": transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2616))
                    ]),
        "cifar100": transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                    ])
    }
    
    test_transforms = {
        "cifar10": transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2616))]),
        "cifar100": transforms.Compose([transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    }

    def __init__(self, seed, client_id, lr, weight_decay, batch_size, momentum, train_data,
                 eval_data, model, dataset, device=None, num_workers=0, run=None):
        self._model = model
        self.id = client_id
        self.train_data = train_data
        self.eval_data = eval_data
        self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers) if self.train_data.__len__() != 0 else None
        self.testloader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers) if self.eval_data.__len__() != 0 else None
        self._classes = self._client_labels()
        self.num_samples_per_class = self.number_of_samples_per_class()
        self.seed = seed
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.run = run
        self.dataset = dataset

    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            num_samples: number of samples used in training
            update: state dictionary of the trained model
        """
        # Train model
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        losses = np.empty(num_epochs)

        for epoch in range(num_epochs):
            self.model.train()
            losses[epoch] = self.run_epoch(optimizer, criterion)

        self.losses = losses
        update = self.model.state_dict()
        return self.num_train_samples, update

    def run_epoch(self, optimizer, criterion):
        """Runs single training epoch of self.model on client's data.

        Return:
            epoch loss
        """
        running_loss = 0.0
        i = 0
        for j, data in enumerate(self.trainloader):
            input_data_tensor, target_data_tensor = data[0].to(self.device), data[1].to(self.device)
            input_data_tensor = self.train_transforms[self.dataset](input_data_tensor)
            optimizer.zero_grad()
            outputs = self.model(input_data_tensor)
            loss = criterion(outputs, target_data_tensor)
            loss.backward()  # gradient inside the optimizer (memory usage increases here)
            running_loss += loss.item()
            optimizer.step()  # update of weights
            i += 1
        if i == 0:
            print("Not running epoch", self.id)
            return 0
        return running_loss / i

    def test(self, batch_size, set_to_use='test'):
        """Tests self.model on self.test_data.

        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            dataloader = self.trainloader
        elif set_to_use == 'test' or set_to_use == 'val':
            dataloader = self.testloader

        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        for data in dataloader:
            input_tensor, labels_tensor = data[0].to(self.device), data[1].to(self.device)
            input_tensor = self.test_transforms[self.dataset](input_tensor)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                test_loss += F.cross_entropy(outputs, labels_tensor, reduction='sum').item()
                _, predicted = torch.max(outputs.data, 1)  # same as torch.argmax()
                total += labels_tensor.size(0)
                correct += (predicted == labels_tensor).sum().item()
        if total == 0:
            accuracy = 0
            test_loss = 0
        else:
            accuracy = 100 * correct / total
            test_loss /= total
        return {ACCURACY_KEY: accuracy, 'loss': test_loss}

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        return self.eval_data.__len__()

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        return self.train_data.__len__()

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        return self.num_train_samples + self.num_test_samples

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model

    def total_grad_norm(self):
        """Returns L2-norm of model total gradient"""
        total_norm = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except Exception:
                    # this param had no grad
                    pass
        total_norm = total_norm ** 0.5
        return total_norm

    def params_norm(self):
        """Returns L2-norm of client's model parameters"""
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def lr_scheduler_step(self, step):
        """Update learning rate according to given step"""
        self.lr *= step

    def update_lr(self, lr):
        self.lr = lr

    def _client_labels(self):
        """Returns client labels (only for analysis purposes)"""
        labels = set()
        if self.train_data.__len__() > 0:
            loader = self.trainloader
        else:
            loader = self.testloader
        for data in loader:
            l = data[1].tolist()
            labels.update(l)
        return list(labels)

    def number_of_samples_per_class(self):
        if self.train_data.__len__() > 0:
            loader = self.trainloader
        else:
            loader = self.testloader
        samples_per_class = {}
        for data in loader:
            labels = data[1].tolist()
            for l in labels:
                if l in samples_per_class:
                    samples_per_class[l] += 1
                else:
                    samples_per_class[l] = 1
        return samples_per_class

    def get_task_info(self):
        """Returns client's task (only for analysis purposes)"""
        return self._classes.copy()