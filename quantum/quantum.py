import copy
import logging
import time

import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
from sklearn.datasets import fetch_lfw_people
from torch.optim import lr_scheduler

from .quantumnet import Quantumnet
from .utils import init_logger, norm_image  # FIXME: Logger is not working
from .utils_quantum import H_layer, RY_layer, entangling_layer


class Quantum():
    def __init__(self, nqubits=32, q_depth=4, q_delta=0.01, max_layers=15, step=0.001, gamma_lr_scheduler=.1,
                 log_file="logs.log", log_level=logging.DEBUG):
        # Init logger
        self.logger = init_logger(log_file, log_level, __name__)
        print("Initializing Quantum class")

        # Init variables
        self.nqubits = nqubits
        self.q_depth = q_depth
        self.q_delta = q_delta
        self.max_layers = max_layers

        self.dataset = {}
        self.dataset_sizes = {}
        self.num_classes = 0

        # Get dataset
        self.get_face_dataset()

        # Init quantum
        self.backend = qml.device('default.qubit', wires=nqubits)

        self.q_net = self._get_q_net_function()

        self.quantumnet = Quantumnet(self.backend, self.q_net, self.nqubits, self.q_delta,
                                     self.max_layers, list(range(self.num_classes)))

        # Init facenet
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running PyTorch on device: {self.device}')

        self.model = InceptionResnetV1(device=self.device)

        # for param in self.model.parameters():  # FIXME: Error with freezed facenet
        #     param.requires_grad = False
        
        self.model.fc = self.quantumnet
        self.model = self.model.to(self.device)

        # Init training variables
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=step)
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=gamma_lr_scheduler)

    def _get_q_net_function(self):
        @qml.qnode(self.backend, interface='torch')
        def q_net(q_in, q_weights_flat):
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

        return q_net

    def get_face_dataset(self):
        """Get faces dataset from sklearn fetch_lfw_people and save it to self.dataset"""
        print("Initializing sklearn fetch_lfw_people dataset")

        # Load data
        lfw_dataset = fetch_lfw_people(min_faces_per_person=100, color=True)

        # Save and split data
        _, h, w, _ = lfw_dataset.images.shape
        x, y = lfw_dataset.images[:-40], lfw_dataset.target[:-40]
        x_val, y_val = lfw_dataset.images[-40:], lfw_dataset.target[-40:]
        self.num_classes = len(lfw_dataset.target_names)

        # Save data so self variables
        self.dataset = {
            "train": [[norm_image(img), label] for img, label in zip(x, y)],
            "val": [[norm_image(img), label] for img, label in zip(x_val, y_val)]
        }
        self.dataset_sizes = {x: len(self.dataset[x]) for x in self.dataset.keys()}

    def train(self, num_epochs, batch_size):
        # Initialize dataloader
        dataloaders = {x: torch.utils.data.DataLoader(self.dataset[x], batch_size=batch_size, shuffle=True,
                                                      num_workers=0) for x in ['train', 'val']}

        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = 10000.0  # Large arbitrary number
        best_acc_train = 0.0
        best_loss_train = 10000.0  # Large arbitrary number
        print('Training started:')
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
                n_batches = self.dataset_sizes[phase] // batch_size
                it = 0
                for x, y in dataloaders[phase]:
                    since_batch = time.time()
                    batch_size_ = len(x)
                    x = x.to(self.device)
                    labels = torch.tensor(y)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()

                    # Track/compute gradient and make an optimization step only when training
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(x)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Print iteration results
                    running_loss += loss.item() * batch_size_
                    batch_corrects = torch.sum(preds == labels.data).item()
                    running_corrects += batch_corrects
                    print('Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}'.format(
                        phase, epoch + 1, num_epochs, it + 1, n_batches + 1, time.time() - since_batch))
                    it += 1

                # Print epoch results
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]
                print('Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}'.format(
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
        print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test loss: {:.4f} | Best test accuracy: {:.4f}'.format(best_loss, best_acc))
        return self.model
