import logging

import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister, BasicAer
from qiskit import execute
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.feature_maps import FirstOrderExpansion, SecondOrderExpansion
from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name

from .all_pairs import AllPairs
from .frqi import c10ry
from .image_preparation import ImagePreparation
from .quantum_edge_detection import quantum_edge_detection as qed
from .swap import swap_12
from .utils import init_logger


class Quantum:
    """Class for interacting with images on IBMQ

    :param face_shape_predict_model: Path to dlib face_shape_predict_model
    :type face_shape_predict_model: str
    :param resize_cover: Images size on IBMQ
    :type resize_cover: (list, tuple)
    :param num_of_shots: Num of shots on IBMQ, use only 2^n numbers
    :type num_of_shots: int
    :param crop_type: Crop type, 0 - No crop, 1 - Crop faces, 2 - Crop faces and use 64 face points
    :type crop_type: int
    :param plt_show: Images size on IBMQ
    :type plt_show: (list, tuple)
    """

    def __init__(self, face_shape_predict_model, resize_cover=(32, 32), num_of_shots=8192, crop_type=0,
                 plt_show=True, log_file="logs.log", log_level=logging.DEBUG):
        self.logger = init_logger(log_file, log_level, __name__)
        self.logger.info("Initializing Quantum class")

        assert crop_type in range(3), "crop_type should be in [0, 1, 2]"

        # Class variables
        self.image_prep = ImagePreparation(face_shape_predict_model)  # TODO: No face_shape_predict_model path
        self.resize_size = resize_cover
        self.numOfShots = num_of_shots
        self.crop_type = crop_type
        self.plt_show = plt_show

        # Dataset variables
        self.dataset = {}
        self.dataset_circuits = {}

        # Initialize qiskit module variables.
        IBMQ.load_account()
        self.provider = IBMQ.get_provider()
        self.backend = self.provider.get_backend('ibmq_qasm_simulator')

        self.logger.debug("Initialized Quantum class")

    def cities_encode(self, norm_images):
        """Generate circuits from images. Method taken from citiesatnight git repository

        :param norm_images: List of images, normalized using ImagePreparation.image_normalization method.
        :type norm_images: list or tuple
        """
        self.logger.info(f"Generating {len(norm_images)} circuits from images")
        circuits = []
        for i, norm_image in enumerate(norm_images):
            # Encode
            anc = QuantumRegister(1, "anc")
            img = QuantumRegister(11, "img")
            anc2 = QuantumRegister(1, "anc2")
            c = ClassicalRegister(12)
            qc = QuantumCircuit(anc, img, anc2, c)

            for i in range(1, len(img)):
                qc.h(img[i])

            # Encode ref image.
            for i in range(len(norm_image)):
                if norm_image[i] != 0:
                    c10ry(qc, 2 * norm_image[i], format(i, '010b'), img[0], anc2[0],
                          [img[j] for j in range(1, len(img))])

            qed(qc)
            qc.measure(anc, c[0])
            qc.measure(img, c[1:12])

            circuits.append(qc)

            self.logger.debug(f"Generated {i}/{len(norm_images)} circuit")

        self.logger.debug(f"Generated {len(norm_images)} circuits from images")
        return circuits

    def generate_images(self, images_path: (list, tuple, str)):
        """Generate images on IBMQ.

        :param images_path: Paths to images, or to dir with them
        :type images_path: [list, tuple, str]
        """
        images, face_images, landmarks, landmarks_images, norm_images = \
            self.image_prep.norm_images_from_disk(images_path, self.crop_type)

        self.logger.info(f"Generating {len(images)} images")

        if self.plt_show:
            if self.crop_type == 0:
                for image in images:
                    plt.imshow(image)
                    plt.show()
            else:
                for image in face_images:
                    plt.imshow(image)
                    plt.show()
                if self.crop_type == 2:
                    for image in landmarks_images:
                        plt.imshow(image)
                        plt.show()

        circuits = self.cities_encode(norm_images)
        generated_images = []
        for image_id, qc, norm_image in enumerate(zip(circuits, norm_images)):
            result = execute(qc, self.backend, shots=self.numOfShots, backend_options={"fusion_enable": True}).result()

            # generated image
            genimg = np.array([])

            # decode
            for i in range(len(norm_image)):
                try:
                    genimg = np.append(genimg, [np.sqrt(result.get_counts(qc)[format(i, '010b') + '10'] /
                                                        self.numOfShots)])
                except KeyError:
                    genimg = np.append(genimg, [0.0])

            # inverse normalization
            genimg *= 32.0 * 255.0  # TODO: Check is 32.0 if connected to self.resize_size

            # convert type
            genimg = genimg.astype('int')

            # back to 2-dim data
            genimg = genimg.reshape(self.resize_size)

            if self.plt_show:
                plt.imshow(genimg, cmap='gray', vmin=0, vmax=255)
                plt.savefig('gen_' + str(image_id) + '.png')
                plt.show()

            generated_images.append(genimg)

        return generated_images

    def swap_compare(self, images_path: (list, tuple, str)):
        """Compare two images on IBMQ using SWAP algorithm.

        :param images_path: Paths to images, or to dir with them
        :type images_path: [list, tuple, str]
        """
        assert len(images_path) == 2, 'Able only to compare two images'
        self.logger.info(f"Comparing 2 images using SWAP algorithm")

        images, face_images, landmarks, landmarks_images, norm_images = \
            self.image_prep.norm_images_from_disk(images_path, self.crop_type)

        target_qubit = QuantumRegister(1, 'target')
        ref = QuantumRegister(11, 'ref')
        original = QuantumRegister(11, 'original')
        anc = QuantumRegister(1, 'anc')
        c = ClassicalRegister(1)

        qc = QuantumCircuit(target_qubit, ref, original, anc, c)

        for i in range(1, len(ref)):
            qc.h(ref[i])

        for i in range(1, len(original)):
            qc.h(original[i])

        # encode ref image
        for i in range(len(norm_images[0])):
            if norm_images[0][i] != 0:
                c10ry(qc, 2 * norm_images[0][i], format(i, '010b'), ref[0], anc[0],
                      [ref[j] for j in range(1, len(ref))])

        # encode original image
        for i in range(len(norm_images[1])):
            if norm_images[1][i] != 0:
                c10ry(qc, 2 * norm_images[1][i], format(i, '010b'), original[0], anc[0],
                      [original[j] for j in range(1, len(original))])

        swap = swap_12(qc, target_qubit, ref, original, c, self.backend, self.numOfShots)

        self.logger.info("Swap results ({} / {}), on images {}, {}".format(*swap.values(), *images_path))
        return swap

    def get_dataset(self):
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
            "train": {i: [] for i in range(n_components)},
            "test": {i: [] for i in range(n_components)},
            "val": {i: [] for i in range(n_components)}
        }
        self.dataset_circuits = {
            "train": {i: [] for i in range(n_components)},
            "test": {i: [] for i in range(n_components)},
            "val": {i: [] for i in range(n_components)}
        }

        for x, y in zip(x_train, y_train):
            self.dataset["train"][y].append(x)
        for x, y in zip(x_test, y_test):
            self.dataset["train"][y].append(x)
        for x, y in zip(x_val, y_val):
            self.dataset["train"][y].append(x)

    def qsvm_train(self, feature_map_type=0):
        """Train QSVM model

        :param feature_map_type: Feature_map type from 0 to 3
        :type feature_map_type: int
        """
        self.logger.info("Preparing QSVM Model dataset")
        # Get dataset if needed
        if not self.dataset:
            self.get_dataset()

        # Normalize dataset images
        for key in self.dataset:
            for i in self.dataset[key]:
                for image in self.dataset[key][i]:
                    self.dataset_circuits[key][i].append(self.image_prep.image_normalization(image, 32, 32))

        # Generate circuits from dataset images
        for key in self.dataset_circuits:
            for i in self.dataset_circuits[key]:
                self.dataset_circuits[key][i] = self.cities_encode(self.dataset_circuits[key][i])

        # Get datapoints from val dataset
        datapoints, class_to_label = split_dataset_to_data_and_labels(self.dataset["val"])

        # Generate feature_map
        if feature_map_type == 0:
            feature_map = FirstOrderExpansion(feature_dimension=len(self.dataset["val"].keys()))
        elif feature_map_type == 1:
            feature_map = RawFeatureVector(feature_dimension=len(self.dataset["val"].keys()))
        elif feature_map_type == 2:
            feature_map = SecondOrderExpansion(feature_dimension=len(self.dataset["val"].keys()))
        elif feature_map_type == 3:
            feature_map = SecondOrderExpansion(feature_dimension=len(self.dataset["val"].keys()), entanglement='linear')
        else:
            raise NotImplementedError

        # Create train QSVM
        qsvm = QSVM(feature_map, self.dataset_circuits["train"], self.dataset_circuits["test"], datapoints[0],
                    multiclass_extension=AllPairs())

        # Create train backend
        # backend = BasicAer.get_backend('qasm_simulator')
        seed = 10675
        quantum_instance = QuantumInstance(self.backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

        # Run train
        self.logger.info("Running QSVM model training")
        result = qsvm.run(quantum_instance)

        self.logger.info(f"Testing success ratio: {result['testing_accuracy']}")
        self.logger.info("Val predictions: "
                         f"Ground truth - {map_label_to_class_name(datapoints[1], qsvm.label_to_class)}, "
                         f"Prediction: {result['predicted_classes']}")

        return result
