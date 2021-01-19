from quantum import Quantum

if __name__ == "__main__":
    q = Quantum("dataset/CelebA", "dataset/labels.txt", batch_size=8, nqubits=32)
    q.train(100)
