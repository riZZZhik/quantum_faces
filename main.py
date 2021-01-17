from quantum import Quantum

if __name__ == "__main__":
    q = Quantum("dataset/CelebA", "dataset/labels.txt", nqubits=32)
    q.train(num_epochs=100)
