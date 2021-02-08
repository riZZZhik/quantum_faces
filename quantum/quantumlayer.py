import numpy as np
import tensorflow as tf
import torch.nn as nn
from keras.layers import Layer
from torch import randn


class QuantumLayer(Layer):
    """Keras quantum layer"""
    def __init__(self, q_net, n_qubits, q_delta, max_layers, **kwargs):
        """Init main class variables.

        :param q_net: Circuit function
        :type q_net: function
        :param nqubits: Number of quantum qubits
        :type nqubits: int
        :param q_delta: Quantum delta
        :type q_delta: float
        :param max_layers: Maximum quantum layers
        :type max_layers: int
        """

        super(QuantumLayer, self).__init__(**kwargs)

        self.n_qubits = n_qubits
        self.q_net = q_net
        self.q_params = nn.Parameter(q_delta * randn(max_layers * n_qubits))  # TODO: Move to keras
        self.print = True

    def get_config(self):
        """Get keras layer config.

        :return: Keras layer config
        """

        config = super(QuantumLayer, self).get_config()
        return config

    def call(self, inputs, **kwargs):
        """Keras layer call function.

        :param inputs: Outputs from previous layer
        :type inputs: list or tuple
        :return: Values after quantum processing
        """

        if tf.executing_eagerly():
            outputs = []
            q_in = tf.tanh(inputs.numpy()) * np.pi / 2.0

            # Apply the quantum circuit to each element of the batch, and append to q_out
            for i, elem in enumerate(q_in):
                q_out_elem = self.q_net(elem, self.q_params).float().unsqueeze(0)
                if self.print:
                    # print(self.q_net.draw())  # FIXME
                    self.q_net.print_applied()
                    self.print = False
                outputs.append(q_out_elem)

            return tf.convert_to_tensor(outputs)
        else:
            return inputs
