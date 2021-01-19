from quantum import Quantum

if __name__ == "__main__":
    q = Quantum(nqubits=32)
    q.train(100, 8, "dataset/CelebA", "dataset/labels.txt")
