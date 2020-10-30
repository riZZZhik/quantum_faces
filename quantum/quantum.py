from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister


class Quantum:
    def __init__(self, resize_cover_size=(32, 32), plt_show=True):
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
