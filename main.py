import os
from datetime import datetime
from glob import glob
from random import sample

from quantum import Quantum


def write_to_file(filename, text):
    if os.path.exists(filename):
        file = open(filename, "a")
        file.write(text)
        file.close()
    else:
        file = open(filename, "w")
        file.write(s)
        file.close()


if __name__ == '__main__':
    files = glob('faces/?.jpg') + glob('faces/?_?.jpg') + glob('faces/*.jpeg') + glob('faces/*.png')

    q = Quantum('quantum_faces/quantum/shape_predictor_68_face_landmarks.dat', crop_type=2)

    # generated_images = q.generate_images(files)

    time = datetime.now().strftime("%H:%M:%S")
    s = "\n\nNew SWAP cycle at {}\n".format(time)
    print(s)
    write_to_file("Results.txt", s)

    i = 0
    while True:
        i += 1

        images = sample(files, 2)
        swap = q.swap_compare(images)

        time = datetime.now().strftime("%H:%M:%S")
        s = "{} Swap {}: {} / {}, on images: {}\n".format(time, i, *swap.values(), images)
        print(s)
        write_to_file("Results.txt", s)
