import numpy as np
import matplotlib.pyplot as plt
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute

from . import quantum_edge_detection as qed
from .utils import norm_images_from_disk


class Quantum:
    def __init__(self, resize_cover_size=(128, 128), plt_show=True):
        self.resize_size = resize_cover_size
        self.plt_show = plt_show

        # Initialize qiskit module variables.
        IBMQ.load_account()
        self.provider = IBMQ.get_provider()
        self.backend = self.provider.backends('ibmq_qasm_simulator')[0]

        self.anc = QuantumRegister(1, "anc")
        self.img = QuantumRegister(11, "img")
        self.anc2 = QuantumRegister(1, "anc2")
        self.c = ClassicalRegister(12)
        self.qc = QuantumCircuit(self.anc, self.img, self.anc2, self.c)

        for i in range(1, len(self.img)):
            self.qc.h(self.img[i])

    def generated_images(self, images_path: (list, tuple, str)):  # TODO: Rename
        # Download and normalize images from disk
        images, norm_images = norm_images_from_disk(images_path, self.resize_size, self.plt_show)

        # TODO: Comments
        qed.quantum_edge_detection(self.qc)
        self.qc.measure(self.anc, self.c[0])
        self.qc.measure(self.img, self.c[1:12])
        print(self.qc.depth())

        numOfShots = 8192
        result = execute(self.qc, self.backend,
                         shots=numOfShots, backend_options={"fusion_enable": True}).result()
        print(result.get_counts(self.qc))

        # TODO: Comments
        generated_images = []
        for image_id, norm_image in enumerate(norm_images):
            # generated image
            genimg = np.array([])

            # decode
            for i in range(len(norm_image)):
                try:
                    genimg = np.append(genimg, [np.sqrt(result.get_counts(self.qc)[format(i, '010b') + '10'] / numOfShots)])
                except KeyError:
                    genimg = np.append(genimg, [0.0])

            # inverse normalization
            genimg *= 32.0 * 255.0

            # convert type
            genimg = genimg.astype('int')

            # back to 2-dim data
            genimg = genimg.reshape((32, 32))

            generated_images.append(genimg)

            if self.plt_show:
                plt.imshow(genimg, cmap='gray', vmin=0, vmax=255)
                plt.show()
                plt.savefig('gen_' + str(image_id) + '.png')

        return generated_images