import logging

import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
from torch.optim import lr_scheduler

from .quantumnet import Quantumnet
from .utils import init_logger
from .utils_quantum import H_layer, RY_layer, entangling_layer


class Quantum():
    def __init__(self, nqubits=32, q_depth=4, q_delta=0.01, max_layers=15, filtered_classes=[0, 1, 2, 3, 4, 5, 6],
                 step=0.001, gamma_lr_scheduler=.1,
                 log_file="logs.log", log_level=logging.DEBUG):
        # Init logger
        self.logger = init_logger(log_file, log_level, __name__)
        self.logger.info("Initializing Quantum class")

        # Init variables
        self.nqubits = nqubits
        self.q_depth = q_depth
        self.q_delta = q_delta
        self.max_layers = max_layers
        self.filtered_classes = filtered_classes

        self.dataset = None
        self.dataset_sizes = None

        # Init quantum
        self.backend = qml.device('default.qubit', wires=nqubits)

        self.q_net = self._get_q_net_function()

        self.quantumnet = Quantumnet(self.backend, self.q_net, self.nqubits, self.q_delta,
                                     self.max_layers, self.filtered_classes)

        # Init facenet
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'Running on device: {self.device}')

        self.model = InceptionResnetV1(pretrained='vggface2')

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = self.quantumnet

        self.model = self.model.to(self.device)

        self.logger.debug("Initialized Quantum class")

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
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import fetch_lfw_people

        self.logger.info("Initializing sklearn fetch_lfw_people dataset")
        # Load data
        lfw_dataset = fetch_lfw_people(min_faces_per_person=100)

        # Save train, test and val datasets to self.dataset
        _, h, w = lfw_dataset.images.shape
        x = lfw_dataset.images[:-40]
        y = lfw_dataset.target[:-40]
        n_components = len(lfw_dataset.target_names)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        x_val, y_val = lfw_dataset.images[-40:], lfw_dataset.target[-40:]

        self.dataset = {
            "train": [x_train, y_train],
            "test": [x_test, y_test],
            "val": [x_val, y_val],
            "n_components": n_components
        }
        self.dataset_sizes = {x: len(self.dataset[x]) for x in self.dataset}
