from quantum import Quantum

if __name__ == "__main__":
    q = Quantum((128, 128, 3), "dataset/CelebA", "dataset/labels.txt", batch_size=16, nqubits=4)
    q.train(100)
