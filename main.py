import os
from datetime import datetime
from glob import glob
from qiskit.providers.ibmq.api_v2.exceptions import RequestsApiError
from random import sample

from quantum import Quantum

files = glob('faces/?.jpg') + glob('faces/?_?.jpg') + glob(
    'faces/*.jpeg') + glob('faces/*.png')
files.sort()


def write_to_file(filename, text):  # TODO: Logger
    if os.path.exists(filename):
        file = open(filename, "a")
        file.write(text)
        file.close()
    else:
        file = open(filename, "w")
        file.write(text)
        file.close()


if __name__ == "__main__":
    q = Quantum('quantum/shape_predictor_68_face_landmarks.dat', crop_type=2)

    q.qsvm_train()

    # i = 0
    # while True:
    #     i += 1
    #     images = sample(files, 2)
    #     time = datetime.now().strftime("%H:%M:%S")
    #
    #     try:
    #         swap = q.swap_compare(images)
    #
    #         if swap:
    #             s = "{} Swap {}: {} / {}, on images: {}".format(time, i, *swap.values(), images)
    #         else:
    #             s = "{} Swap {}: No faces found, on images: {}".format(time, i, images)
    #         print(s)
    #         write_to_file("Results.txt", s + "\n")
    #
    #     except RequestsApiError:
    #         s = "{} Swap {}: Requests API Error, on images: {}".format(time, i, images)
    #         print(s)
    #         write_to_file("Results.txt", s + "\n")

