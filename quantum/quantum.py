import sys

import pennylane as qml
from keras import Model
from keras.applications import Xception
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Softmax
from loguru import logger

from .generator import Generator
from .quantumlayer import QuantumLayer
from .utils_quantum import H_layer, RY_layer, entangling_layer


class Quantum:
    """Quantum Transfer Learning model class"""
    def __init__(self, batch_size, image_shape, images_dir, labels_path, label_max_filter=None,
                 face_shape_predict_model=None,
                 nqubits=32, q_depth=4, q_delta=0.01, max_layers=15, log_file="logs.log"):
        """Init main class variables.

        :param batch_size: Batch size
        :type batch_size: int
        :param image_shape: Image shape
        :type image_shape: list or tuple
        :param images_dir: Path to images dir
        :type images_dir: str
        :param labels_path: Path to labels file
        :type labels_path: str
        :param label_max_filter: Max label id to filter images
        :type label_max_filter: int
        :param face_shape_predict_model: Path to dlib face_shape_predict_model
        :type face_shape_predict_model: str
        :param nqubits: Number of quantum qubits
        :type nqubits: int
        :param q_depth: Quantum depth
        :type q_depth: int
        :param q_delta: Quantum delta
        :type q_delta: float
        :param max_layers: Maximum quantum layers
        :type max_layers: int
        :param log_file: Path to log file
        :type log_file: str
        """

        # Init logger
        if log_file:
            logger.add(log_file)
        logger.info("Initializing Quantum class")

        # Init variables
        self.batch_size = batch_size
        self.image_shape = image_shape

        self.nqubits = nqubits
        self.q_depth = q_depth
        self.q_delta = q_delta
        self.max_layers = max_layers

        freeze_imagenet = False

        # Init dataset
        logger.info("Initializing dataset")
        self.train_generator = Generator(batch_size, image_shape, images_dir, labels_path, label_max_filter,
                                         face_shape_predict_model)

        # Init quantum
        try:
            self.backend = qml.device('default.qubit', wires=nqubits)
        except MemoryError as e:
            logger.critical(e)
            logger.critical('Try to change overcommit_memory settings with command:\n'
                            'sudo sh -c "/usr/bin/echo 1 > /proc/sys/vm/overcommit_memory"')
            sys.exit(-1)

        # Initialize XCeption + QuantumNet model
        base_model = Xception(input_shape=self.image_shape, weights='imagenet', include_top=False)

        x = base_model.output
        x = Flatten()(x)
        x = Dense(nqubits)(x)
        x = QuantumLayer(self._get_q_net_function(), self.nqubits, self.q_delta, self.max_layers)(x)
        x = Dense(self.train_generator.num_classes)(x)
        predictions = Softmax()(x)

        self.model = Model(base_model.input, predictions)

        if freeze_imagenet:
            for layer in base_model.layers:
                layer.trainable = False

        # Compile model
        self.model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['accuracy'])
        # self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=gamma_lr_scheduler)  # TODO: do we need this in keras?

    def _get_q_net_function(self):
        """Get circuit function.

        :return: Circuit function
        """

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

    def train(self, num_epochs):
        """Train Keras Transfer Learning model.

        :param num_epochs: Number of epochs
        :type num_epochs: int
        """

        callbacks = [ModelCheckpoint(filepath='checkpoints/model.{epoch:02d}-{accuracy:.2f}.h5', monitor="accuracy")]
        self.model.fit(self.train_generator, epochs=num_epochs, callbacks=callbacks)
