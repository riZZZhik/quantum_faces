import numpy as np

import torch
import torch.nn as nn


class Quantumnet(nn.Module):
    def __init__(self, device, q_net, n_qubits, q_delta, max_layers, filtered_classes):
        super().__init__()

        self.device = device
        self.q_net = q_net
        self.n_qubits = n_qubits
        self.q_delta = q_delta
        self.max_layers = max_layers
        self.filtered_classes = filtered_classes

        self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(max_layers * n_qubits))
        self.post_net = nn.Linear(n_qubits, len(filtered_classes))

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch, and append to q_out
        q_out = torch.Tensor(0, self.n_qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = self.q_net(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return self.post_net(q_out)
