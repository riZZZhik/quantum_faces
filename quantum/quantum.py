import numpy as np
import matplotlib.pyplot as plt
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit import execute

from .frqi import c10ry
from . import quantum_edge_detection as qed
from .utils import norm_images_from_disk, norm_face_images_from_disk


class Quantum:
    def __init__(self, ref_image_path, resize_cover_size=(32, 32), crop_faces=True, plt_show=True):
        self.resize_size = resize_cover_size
        self.crop_faces = crop_faces
        self.plt_show = plt_show

        # Initialize qiskit module variables.
        IBMQ.load_account()
        self.provider = IBMQ.get_provider()
        self.backend = self.provider.get_backend('ibmq_qasm_simulator')

        self.anc = QuantumRegister(1, "anc")
        self.img = QuantumRegister(11, "img")
        self.anc2 = QuantumRegister(1, "anc2")
        self.c = ClassicalRegister(12)
        self.qc = QuantumCircuit(self.anc, self.img, self.anc2, self.c)

        for i in range(1, len(self.img)):
            self.qc.h(self.img[i])

        # Encode ref image.
        if self.crop_faces:
            norm_image = norm_face_images_from_disk([ref_image_path], self.resize_size)[2][0]
        else:
            norm_image = norm_images_from_disk([ref_image_path], self.resize_size)[1][0]

        for i in range(len(norm_image)):
            if norm_image[i] != 0:
                c10ry(self.qc, 2 * norm_image[i], format(i, '010b'), self.img[0], self.anc2[0],
                      [self.img[j] for j in range(1, len(self.img))])

        qed.quantum_edge_detection(self.qc)
        self.qc.measure(self.anc, self.c[0])
        self.qc.measure(self.img, self.c[1:12])
        print(self.qc.depth())

        self.numOfShots = 8192
        self.result = execute(self.qc, self.backend,
                              shots=self.numOfShots, backend_options={"fusion_enable": True}).result()

        circuit_drawer(self.qc).show()
        plot_histogram(self.result.get_counts(self.qc))
        print(self.result.get_counts(self.qc))

    def generated_images(self, images_path: (list, tuple, str)):  # TODO: Rename
        # Download and normalize images from disk
        if self.crop_faces:
            images, face_images, norm_images = norm_face_images_from_disk(images_path, self.resize_size, self.plt_show)
        else:
            images, norm_images = norm_images_from_disk(images_path, self.resize_size, self.plt_show)

        # TODO: Comments
        generated_images = []
        for image_id, norm_image in enumerate(norm_images):
            # generated image
            genimg = np.array([])

            # decode
            for i in range(len(norm_image)):
                try:
                    genimg = np.append(genimg, [np.sqrt(self.result.get_counts(self.qc)[format(i, '010b') + '10'] /
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

            generated_images.append(genimg)

        return generated_images
