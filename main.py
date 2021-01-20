from quantum import Quantum

if __name__ == "__main__":
    q = Quantum("dataset/CelebA", "dataset/labels.txt", batch_size=16, nqubits=4,
                dataset_delta=1, label_max_filter=100)
    q.train(100)
