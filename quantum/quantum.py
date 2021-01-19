import copy
import sys
import time

import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
from loguru import logger
from torch.optim import lr_scheduler

from .quantumnet import Quantumnet
from .utils import get_celeba_generator
from .utils_quantum import H_layer, RY_layer, entangling_layer


class Quantum:  # TODO: Comments
    def __init__(self, nqubits=32, q_depth=4, q_delta=0.01, max_layers=15, step=0.001, gamma_lr_scheduler=.1,
                 log_file="logs.log"):
        # Init logger
        if log_file:
            logger.add(log_file)
        logger.info("Initializing Quantum class")

        # Init variables
        self.nqubits = nqubits
        self.q_depth = q_depth
        self.q_delta = q_delta
        self.max_layers = max_layers

        # Init torch device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Running PyTorch on device: {self.device}')

        # Init quantum
        try:
            self.backend = qml.device('default.qubit', wires=nqubits)
        except MemoryError as e:
            logger.critical(e)
            logger.critical('Try to change overcommit_memory settings with command:\n'
                            'sudo sh -c "/usr/bin/echo 1 > /proc/sys/vm/overcommit_memory"')
            sys.exit(-1)

        self.q_net = self._get_q_net_function()
        self.quantumnet = Quantumnet(self.device, self.q_net, self.nqubits, self.q_delta,
                                     self.max_layers, list(range(self.num_classes)))

        # Init facenet
        self.model = InceptionResnetV1(classify=True, num_classes=nqubits)
        self.model.logits = self.quantumnet

        # for param in self.model.parameters():  # FIXME: Error with frozen facenet
        #     param.requires_grad = False
        self.model = self.model.to(self.device)

        # Init training variables
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.logits.parameters(), lr=step)
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=gamma_lr_scheduler)

    def _get_q_net_function(self):
        @qml.qnode(self.backend, interface='torch')
        def q_net_circuit(q_in, q_weights_flat):
            # Reshape weights
            q_weights = q_weights_flat.reshape(self.max_layers, self.nqubits)

            # Start from state |+> , unbiased w.r.t. |0> and |1>
            H_layer(self.nqubits)

            # Embed features in the quantum node
            RY_layer(q_in)

            # Sequence of trainable variational layers
            for k in range(self.q_depth):
                entangling_layer(self.nqubits)
                RY_layer(q_weights[k + 1])

            # Expectation values in the Z basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.nqubits)]

        return q_net_circuit

    def train(self, num_epochs, batch_size, images_dir, labels_path):
        # Init dataset
        self.dataset_generators, self.dataset_sizes, self.num_classes = \
            get_celeba_generator(batch_size, 40)

        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = 10000.0  # Large arbitrary number
        best_acc_train = 0.0
        best_loss_train = 10000.0  # Large arbitrary number
        logger.info('Training started:')
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    # Set model to training mode
                    self.exp_lr_scheduler.step()
                    self.model.train()
                else:
                    # Set model to evaluate mode
                    self.model.eval()

                    # Iteration loop
                running_loss = 0.0
                running_corrects = 0
                n_batches = self.dataset_sizes[phase] // self.nqubits
                it = 0
                for x, y in self.dataset_generators[phase]():
                    since_batch = time.time()
                    batch_size_ = len(x)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()

                    # Track/compute gradient and make an optimization step only when training
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(x)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, y)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Print iteration results
                    running_loss += loss.item() * batch_size_
                    batch_corrects = torch.sum(preds == y.data).item()
                    running_corrects += batch_corrects
                    logger.info('Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}'.format(
                        phase, epoch + 1, num_epochs, it + 1, n_batches + 1, time.time() - since_batch))
                    it += 1

                # Print epoch results
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]
                logger.info('Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}'.format(
                    'train' if phase == 'train' else 'val  ', epoch + 1, num_epochs, epoch_loss, epoch_acc))

                # Check if this is the best model wrt previous epochs
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                if phase == 'train' and epoch_acc > best_acc_train:
                    best_acc_train = epoch_acc
                if phase == 'train' and epoch_loss < best_loss_train:
                    best_loss_train = epoch_loss

        # Print final results
        self.model.load_state_dict(best_model_wts)
        time_elapsed = time.time() - since
        logger.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Best test loss: {:.4f} | Best test accuracy: {:.4f}'.format(best_loss, best_acc))
        return self.model
