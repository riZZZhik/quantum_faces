from datetime import datetime
from glob import glob
from random import sample

from quantum import Quantum


def write_to_file(filename, text):
    file = open(filename, "a")
    file.write(text)
    file.close()


if __name__ == '__main__':
    files = glob('faces/?.jpg') + glob('faces/?_?.jpg') + glob('faces/*.jpeg') + glob('faces/*.png')

    q = Quantum(crop_type=2)

    generated_images = q.generate_images(files)

    # time = datetime.now().strftime("%H:%M:%S")
    # write_to_file("Results.txt", "\n\nNew SWAP cycle at {}\n".format(time))
    #
    # i = 0
    # while True:
    #     i += 1
    #
    #     images = sample(files, 2)
    #     swap = q.swap_compare(images)
    #
    #     time = datetime.now().strftime("%H:%M:%S")
    #     s = "{} Swap {}: {} / {}, on images: {}\n".format(time, i, *swap.values(), images)
    #     print(s)
    #     write_to_file("Results.txt", s)
