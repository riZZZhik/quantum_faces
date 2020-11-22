import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute

from .frqi import c10ry
from .quantum_edge_detection import quantum_edge_detection as qed
from .swap import swap_12
from .image_preparation import ImagePreparation


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
                 plt_show=True):
        assert crop_type in range(3), "crop_type should be in [0, 1, 2]"

        self.image_prep = ImagePreparation(face_shape_predict_model)
        self.resize_size = resize_cover
        self.numOfShots = num_of_shots
        self.crop_type = crop_type
        self.plt_show = plt_show

        # Initialize qiskit module variables.
        IBMQ.load_account()
        self.provider = IBMQ.get_provider()
        self.backend = self.provider.get_backend('ibmq_qasm_simulator')

    def generate_images(self, images_path: (list, tuple, str)):
        """Generate images on IBMQ.

        :param images_path: Paths to images, or to dir with them
        :type images_path: [list, tuple, str]
        """
        images, face_images, landmarks, landmarks_images, norm_images = \
            self.image_prep.norm_images_from_disk(images_path, self.crop_type)

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

        generated_images = []
        for image_id, norm_image in enumerate(norm_images):
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
            print(qc.depth())

            result = execute(qc, self.backend, shots=self.numOfShots, backend_options={"fusion_enable": True}).result()

            print(result.get_counts(qc))

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

        images, face_images, landmarks, landmarks_images, norm_images = \
            self.image_prep.norm_images_from_disk(images_path, self.crop_type)

        if len(norm_images) == 2:
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

            return swap_12(qc, target_qubit, ref, original, c, self.backend, self.numOfShots)
