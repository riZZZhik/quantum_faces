from quantum import Quantum

if __name__ == "__main__":
    q = Quantum((71, 71, 3), "dataset/CelebA", "dataset/labels.txt", batch_size=16, nqubits=4,
                face_shape_predict_model="quantum/shape_predictor_68_face_landmarks.dat")
    q.train(100)
