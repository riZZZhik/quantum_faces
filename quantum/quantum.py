import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute

from .frqi import c10ry
from .quantum_edge_detection import quantum_edge_detection as qed
from .utils import norm_images_from_disk, norm_face_images_from_disk


class Quantum:
    def __init__(self, resize_cover_size=(32, 32), num_of_shots=8192, crop_faces=True, plt_show=True):
        self.resize_size = resize_cover_size
        self.numOfShots = num_of_shots
        self.crop_faces = crop_faces
        self.plt_show = plt_show

        # Initialize qiskit module variables.
        IBMQ.load_account()
        self.provider = IBMQ.get_provider()
        self.backend = self.provider.get_backend('ibmq_qasm_simulator')

    def generate_images(self, images_path: (list, tuple, str)):  # TODO: Rename
        # Download and normalize images from disk
        if self.crop_faces:
            images, face_images, norm_images = norm_face_images_from_disk(images_path, self.resize_size, self.plt_show)
        else:
            images, norm_images = norm_images_from_disk(images_path, self.resize_size, self.plt_show)

        # TODO: Comments
        generate_images = []
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
            genimg *= 32.0 * 255.0  # FIXME: Check is 32.0 if connected to self.resize_size

            # convert type
            genimg = genimg.astype('int')

            # back to 2-dim data
            genimg = genimg.reshape(self.resize_size)

            if self.plt_show:
                plt.imshow(genimg, cmap='gray', vmin=0, vmax=255)
                plt.savefig('gen_' + str(image_id) + '.png')
                plt.show()

            generate_images.append(genimg)

        return generate_images
