from quantum import Quantum


if __name__ == "__main__":
    q = Quantum('quantum/shape_predictor_68_face_landmarks.dat', crop_type=2, log_file="logs.log")

    q.qsvm_train()

    # files = glob('faces/?.jpg') + glob('faces/?_?.jpg') + glob(
    #     'faces/*.jpeg') + glob('faces/*.png')
    # files.sort()

    # i = 0
    # while True:
    #     i += 1
    #     images = sample(files, 2)
    #     swap = q.swap_compare(images)

